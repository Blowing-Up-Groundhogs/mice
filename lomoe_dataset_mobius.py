"""
Dataset loader for the SAM3-masked LoMOE-Bench dataset.
Reads from a dataset root directory that uses the same structure as the Mobius-masked dataset
(LoMOE_mobius.json with updated mask paths pointing to SAM3 masks).
"""
import json
import re
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from loguru import logger

# Original benchmark root (for prompt text files only)
LOMOE_BENCH_ORIG = Path("/mnt/ssd1/LoMOE/benchmark/data/LoMOE-Bench")

DEFAULT_ROOT = Path("/mnt/ssd1/lomoe_bench_mobius")


def parse_quoted_strings(line: str) -> list:
    return re.findall(r'"([^"]*)"', line)


def load_mask(mask_path: Path) -> torch.Tensor:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask = torch.from_numpy(np.array(Image.open(mask_path).convert('L')) > 127).float()
    return mask


def get_bbox_from_mask(mask_path: Path) -> list:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask_np = np.array(Image.open(mask_path).convert('L'))
    y_coords, x_coords = np.where(mask_np > 127)
    if len(x_coords) == 0:
        raise ValueError(f"Empty mask: {mask_path}")
    x1, x2 = x_coords.min(), x_coords.max()
    y1, y2 = y_coords.min(), y_coords.max()
    h, w = mask_np.shape
    return [float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h]


def load_prompts_data(lomoe_config: dict) -> dict:
    source_file = LOMOE_BENCH_ORIG / 'utils/mask_orig_prompts.txt'
    target_file = LOMOE_BENCH_ORIG / 'utils/text_prompts.txt'

    if not source_file.exists() or not target_file.exists():
        logger.warning("Prompt text files not found. Falling back to JSON fg_prompt.")
        prompts_data = {}
        for sample_id, config in lomoe_config.items():
            prompts_data[sample_id] = {
                'sources': [],
                'targets': parse_quoted_strings(config.get('fg_prompt', "")),
                'image_path': config['image_path'],
                'mask_paths': parse_quoted_strings(config.get('mask_path', "")),
                'edit_instance_single': config.get('edit_inst_single', ''),
            }
        return prompts_data

    with open(source_file, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()
    with open(target_file, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()

    sorted_sample_ids = sorted(lomoe_config.keys(), key=lambda x: int(x))

    if len(sorted_sample_ids) != len(source_lines):
        logger.warning(
            f"Mismatch: {len(sorted_sample_ids)} JSON samples vs {len(source_lines)} txt lines. "
            f"Truncating to minimum."
        )

    prompts_data = {}
    for sample_id, src_line, tgt_line in zip(sorted_sample_ids, source_lines, target_lines):
        sources = parse_quoted_strings(src_line)
        targets = parse_quoted_strings(tgt_line)
        prompts_data[sample_id] = {
            'sources': sources,
            'targets': targets,
            'image_path': lomoe_config[sample_id]['image_path'],
            'mask_paths': parse_quoted_strings(lomoe_config[sample_id]['mask_path']),
            'edit_instance_single': lomoe_config[sample_id].get('edit_inst_single', ''),
        }

    return prompts_data


class LoMOEMobiusDataset(Dataset):
    def __init__(self, root_dir=None, target_size=1024, vae_scale_factor=8):
        self.target_size = target_size
        self.vae_scale_factor = vae_scale_factor
        self.root_dir = Path(root_dir) if root_dir else DEFAULT_ROOT

        config_path = self.root_dir / 'LoMOE_mobius.json'
        if not config_path.exists():
            raise FileNotFoundError(
                f"LoMOE_mobius.json not found at {config_path}."
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            self.lomoe_config = json.load(f)

        self.prompts_data = load_prompts_data(self.lomoe_config)
        self.sorted_ids = sorted(self.prompts_data.keys(), key=lambda x: int(x))

    def __len__(self):
        return len(self.sorted_ids)

    def __getitem__(self, idx):
        sample_id = self.sorted_ids[idx]
        sample_prompts = self.prompts_data[sample_id]

        image_path = self.root_dir / sample_prompts['image_path']
        if not image_path.exists():
            image_path = LOMOE_BENCH_ORIG / sample_prompts['image_path']
        source_image = Image.open(image_path).convert('RGB')

        instance_bboxes = []
        instance_masks = []
        for mask_rel in sample_prompts['mask_paths']:
            mask_path = self.root_dir / mask_rel
            if not mask_path.exists() or 'MISSING_MASK' in mask_rel:
                logger.warning(f"Mask missing for sample {sample_id}: {mask_rel} — using zero mask")
                h, w = source_image.size[1], source_image.size[0]
                instance_masks.append(torch.zeros(h, w))
                instance_bboxes.append([0.0, 0.0, 1.0, 1.0])
                continue
            try:
                instance_masks.append(load_mask(mask_path))
                instance_bboxes.append(get_bbox_from_mask(mask_path))
            except Exception as e:
                logger.warning(f"Failed to load mask {mask_path}: {e} — using zero mask")
                h, w = source_image.size[1], source_image.size[0]
                instance_masks.append(torch.zeros(h, w))
                instance_bboxes.append([0.0, 0.0, 1.0, 1.0])

        text_sources = sample_prompts['sources']
        text_targets = sample_prompts['targets']

        global_prompt = " "
        prompts = [global_prompt]
        for src, tgt in zip(text_sources, text_targets):
            prompts.append(f'Replace {src} with {tgt}')
        prompt_with_breakflag = '$BREAKFLAG$'.join(prompts)

        original_width, original_height = source_image.size
        aspect_ratio = original_width / original_height
        width = round((self.target_size * self.target_size * aspect_ratio) ** 0.5)
        height = round((self.target_size * self.target_size / aspect_ratio) ** 0.5)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        source_image_resized = source_image.resize((width, height), resample=Image.LANCZOS)

        return {
            "image": source_image_resized,
            "prompt": prompt_with_breakflag,
            "edit_instance_single": sample_prompts['edit_instance_single'],
            "bboxes": instance_bboxes,
            "masks": instance_masks,
            "sample_id": sample_id,
            "original_size": [original_width, original_height],
        }


def collate_fn(batch):
    return batch


def get_lomoe_mobius_dataloader(root_dir=None, batch_size=1, shuffle=False, num_workers=0, **kwargs):
    dataset = LoMOEMobiusDataset(root_dir=root_dir, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
