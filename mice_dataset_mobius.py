import json
import re
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def parse_quoted_strings(line: str) -> list[str]:
    if not line:
        return []
    return re.findall(r'"([^"]*)"', line)

def load_mask(mask_path: Path) -> torch.Tensor:
    if not mask_path.exists():
        print(f"Warning: Mask path {mask_path} does not exist. Using empty mask.")
        return torch.zeros((512, 512))
    mask = torch.from_numpy(np.array(Image.open(mask_path).convert('L')) > 127).float()
    return mask

def get_bbox_from_mask(mask_path: Path) -> list[float]:
    if not mask_path.exists():
        print(f"Warning: Mask path {mask_path} does not exist. Using full image bbox.")
        return [0.0, 0.0, 1.0, 1.0]
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    y_coords, x_coords = np.where(mask_np > 127)
    if len(x_coords) == 0:
        print(f"Warning: Mask {mask_path} is empty. Using full image bbox.")
        return [0.0, 0.0, 1.0, 1.0]
    x1, x2 = x_coords.min(), x_coords.max()
    y1, y2 = y_coords.min(), y_coords.max()
    h, w = mask_np.shape
    return [float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h]


class MiceBenchMobiusDataset(Dataset):
    def __init__(self, root_dir='/mnt/ssd1/mice_bench_mobius', target_size=1024, vae_scale_factor=8):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.vae_scale_factor = vae_scale_factor

        config_path = self.root_dir / 'LoMOE_mobius.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            self.lomoe_config = json.load(f)

        self.sorted_ids = sorted(self.lomoe_config.keys(), key=lambda x: int(x))

    def __len__(self):
        return len(self.sorted_ids)

    def __getitem__(self, idx):
        sample_id = self.sorted_ids[idx]
        sample_data = self.lomoe_config[sample_id]

        image_path = self.root_dir / sample_data['image_path']
        if not image_path.exists():
            if str(image_path).endswith('.jpg'):
                image_path = Path(str(image_path)[:-4] + '.png')
            elif str(image_path).endswith('.png'):
                image_path = Path(str(image_path)[:-4] + '.jpg')

        if not image_path.exists():
            raise FileNotFoundError(f"Image path {image_path} does not exist.")

        source_image = Image.open(image_path).convert('RGB')

        mask_paths = parse_quoted_strings(sample_data.get('mask_path', ""))

        instance_bboxes = []
        for mask_p in mask_paths:
            bbox = get_bbox_from_mask(self.root_dir / mask_p)
            instance_bboxes.append(bbox)

        instance_masks = []
        for mask_p in mask_paths:
            mask = load_mask(self.root_dir / mask_p)
            instance_masks.append(mask)

        text_sources = parse_quoted_strings(sample_data.get('source_prompt', ""))
        text_targets = parse_quoted_strings(sample_data.get('fg_prompt', ""))

        global_prompt = " "
        prompts = [global_prompt]
        for src, tgt in zip(text_sources, text_targets):
            prompts.append(f'Replace the {src} with {tgt}')

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
            "sources": text_sources,
            "targets": text_targets,
            "bboxes": instance_bboxes,
            "masks": instance_masks,
            "sample_id": sample_id,
            "original_size": [original_width, original_height],
            'multi_turn_enhanced': sample_data.get('edit_inst_multi', ''),
            'edit_instance_single': sample_data.get('edit_inst_single', ''),
        }

def collate_fn(batch):
    return batch

def get_mice_mobius_dataloader(root_dir='/mnt/ssd1/mice_bench_mobius', batch_size=1, shuffle=False, num_workers=0, **kwargs):
    dataset = MiceBenchMobiusDataset(root_dir=root_dir, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
