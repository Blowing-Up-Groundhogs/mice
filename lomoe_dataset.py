import json
import re
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from loguru import logger

LOMOE_BASE_PATH = Path(__file__).parent
LOMOE_BENCH_PATH = LOMOE_BASE_PATH / 'benchmark/data/LoMOE-Bench'

def parse_quoted_strings(line: str) -> list[str]:
    return re.findall(r'"([^"]*)"', line)


def load_mask(mask_path: Path) -> torch.Tensor:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask path {mask_path} does not exist.")
    mask = torch.from_numpy(np.array(Image.open(mask_path).convert('L')) > 127).float()
    return mask


def get_bbox_from_mask(mask_path: Path) -> list[float]:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask path {mask_path} does not exist.")
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    y_coords, x_coords = np.where(mask_np > 127)
    if len(x_coords) == 0:
        raise ValueError(f"Mask {mask_path} is empty.")
    x1, x2 = x_coords.min(), x_coords.max()
    y1, y2 = y_coords.min(), y_coords.max()
    h, w = mask_np.shape
    return [float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h]


def load_prompts_data(lomoe_config: dict, multi_turn_faithful_config: dict, multi_turn_enhanced_config: dict) -> dict[str, dict]:
    source_file = LOMOE_BENCH_PATH / 'utils/mask_orig_prompts.txt'
    target_file = LOMOE_BENCH_PATH / 'utils/text_prompts.txt'

    if not source_file.exists() or not target_file.exists():
        logger.warning("Prompt text files not found. Falling back to JSON config.")
        prompts_data = {}
        for sample_id, config in lomoe_config.items():
            prompts_data[sample_id] = {
                'sources': [],
                'targets': parse_quoted_strings(config.get('fg_prompt', "")),
                'image_path': config['image_path'],
                'mask_paths': parse_quoted_strings(config.get('mask_path', "")),
            }
        return prompts_data

    with open(source_file, 'r', encoding='utf-8') as f:
        source_lines = f.readlines()

    with open(target_file, 'r', encoding='utf-8') as f:
        target_lines = f.readlines()

    sorted_sample_ids = sorted(lomoe_config.keys(), key=lambda x: int(x))

    if len(sorted_sample_ids) != len(source_lines):
        logger.warning(f"Mismatch: {len(sorted_sample_ids)} JSON samples vs {len(source_lines)} txt lines. Truncating to minimum.")

    prompts_data = {}
    for sample_id, src_line, tgt_line in zip(sorted_sample_ids, source_lines, target_lines):
        sources = parse_quoted_strings(src_line)
        targets = parse_quoted_strings(tgt_line)
        prompts_data[sample_id] = {
            'sources': sources,
            'targets': targets,
            'image_path': lomoe_config[sample_id]['image_path'],
            'mask_paths': parse_quoted_strings(lomoe_config[sample_id]['mask_path']),
            "edit_instance_single": lomoe_config[sample_id]['edit_inst_single'],
            "multi_turn_faithful": multi_turn_faithful_config[sample_id],
            "multi_turn_enhanced": multi_turn_enhanced_config[sample_id],
        }

    return prompts_data

class LoMOEDataset(Dataset):
    def __init__(self, target_size=1024, vae_scale_factor=8):
        self.target_size = target_size
        self.vae_scale_factor = vae_scale_factor

        config_path = LOMOE_BENCH_PATH / 'LoMOE.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            self.lomoe_config = json.load(f)

        multi_turn_faithful_path = LOMOE_BENCH_PATH / 'LoMOE_multi_turn_faithful.json'
        with open(multi_turn_faithful_path, 'r', encoding='utf-8') as f:
            self.multi_turn_faithful_config = json.load(f)

        multi_turn_enhanced_path = LOMOE_BENCH_PATH / 'LoMOE_multi_turn_enhanced.json'
        with open(multi_turn_enhanced_path, 'r', encoding='utf-8') as f:
            self.multi_turn_enhanced_config = json.load(f)

        self.prompts_data = load_prompts_data(self.lomoe_config, self.multi_turn_faithful_config, self.multi_turn_enhanced_config)
        self.sorted_ids = sorted(self.prompts_data.keys(), key=lambda x: int(x))

    def __len__(self):
        return len(self.sorted_ids)

    def __getitem__(self, idx):
        sample_id = self.sorted_ids[idx]
        sample_prompts = self.prompts_data[sample_id]

        image_path = LOMOE_BENCH_PATH / sample_prompts['image_path']
        source_image = Image.open(image_path).convert('RGB')

        instance_bboxes = []
        for mask_p in sample_prompts['mask_paths']:
            bbox = get_bbox_from_mask(LOMOE_BENCH_PATH / mask_p)
            instance_bboxes.append(bbox)

        instance_masks = []
        for mask_p in sample_prompts['mask_paths']:
            mask = load_mask(LOMOE_BENCH_PATH / mask_p)
            instance_masks.append(mask)

        text_sources = sample_prompts['sources']
        text_targets = sample_prompts['targets']

        global_prompt = " "
        prompts = [global_prompt]
        for src, tgt in zip(text_sources, text_targets):
            prompts.append(f'Replace {src} with {tgt}')

        prompt_2 = '$BREAKFLAG$'.join(prompts)

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
            "prompt": prompt_2,
            "sources": text_sources,
            "edit_instance_single": sample_prompts['edit_instance_single'],
            "bboxes": instance_bboxes,
            "masks": instance_masks,
            "sample_id": sample_id,
            "original_size": [original_width, original_height],
            'targets': text_targets,
            "multi_turn_faithful": sample_prompts['multi_turn_faithful'],
            "multi_turn_enhanced": sample_prompts['multi_turn_enhanced'],
        }


def collate_fn(batch):
    return batch


def get_lomoe_dataloader(batch_size=1, shuffle=False, num_workers=0, **kwargs):
    dataset = LoMOEDataset(**kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
