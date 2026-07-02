# MICE

Code for spatially-controlled multi-instance image editing with FLUX.2-klein (4B), as evaluated on MICE-Bench and LoMOE-Bench.

## Requirements

```bash
pip install torch diffusers transformers pillow tqdm loguru numpy
```

The model will be downloaded automatically from HuggingFace on first run:
```
black-forest-labs/FLUX.2-klein-4B
```

## Dataset Setup

### MICE-Bench

Download MICE-Bench and place it so the directory contains:
```
mice_bench/
  LoMOE.json
  images/
  masks/
```

Pass the path with `--dataset_root`.

### LoMOE-Bench

LoMOE-Bench is expected at `benchmark/data/LoMOE-Bench/` relative to this repository root (this path is hardcoded in `lomoe_dataset.py`). The directory must contain:
```
benchmark/data/LoMOE-Bench/
  LoMOE.json
  LoMOE_multi_turn_faithful.json
  LoMOE_multi_turn_enhanced.json
  utils/
    mask_orig_prompts.txt
    text_prompts.txt
  images/
  masks/
```

### SAM3 variants

The `_sam3` scripts expect a dataset root whose `LoMOE_mobius.json` has mask paths pointing to SAM3-generated masks. Pass the path with `--dataset_root`.

## Running Inference

### MICE-Bench (ground-truth masks)

```bash
python infer_flux2_mice.py \
  --exp_name my_experiment \
  --dataset_root ../mice_bench \
  --num_inference_steps 4 \
  --kernel_size 11 \
  --temperature 3.0 \
  --hard_image_attribute_binding_list_double 0,5 \
  --hard_image_attribute_binding_list_single 0,20 \
  --use_masks
```

Results are saved to `results_micebench/mice_flux2_klein_<exp_name>/`.

### MICE-Bench (SAM3 masks)

```bash
python infer_flux2_mice_sam3.py \
  --exp_name my_experiment \
  --dataset_root /path/to/mice_bench_sam3 \
  --num_inference_steps 4 \
  --kernel_size 11 \
  --temperature 3.0 \
  --use_masks
```

Results are saved to `results_micebench_sam3/mice_flux2_sam3_<exp_name>/`.

### LoMOE-Bench (ground-truth masks)

```bash
python infer_flux2_lomoe.py \
  --exp_name my_experiment \
  --num_inference_steps 4 \
  --kernel_size 11 \
  --temperature 3.0 \
  --hard_image_attribute_binding_list_double 0,5 \
  --hard_image_attribute_binding_list_single 0,20 \
  --use_masks
```

Results are saved to `results/lomoe_flux2_klein_<exp_name>/`.

### LoMOE-Bench (SAM3 masks)

```bash
python infer_flux2_lomoe_sam3.py \
  --exp_name my_experiment \
  --dataset_root /path/to/lomoe_bench_sam3 \
  --num_inference_steps 4 \
  --kernel_size 11 \
  --temperature 3.0 \
  --use_masks
```

Results are saved to `results_sam3/lomoe_flux2_sam3_<exp_name>/`.

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--num_inference_steps` | `4` | Number of denoising steps |
| `--kernel_size` | `11` | Spatial kernel size for attention falloff |
| `--temperature` | `3.0` | Controls steepness of spatial falloff |
| `--use_masks` | flag | Use segmentation masks; omit to use bounding boxes instead |
| `--hard_image_attribute_binding_list_double` | `0,5` | Range of double-stream blocks to apply attention binding |
| `--hard_image_attribute_binding_list_single` | `0,20` | Range of single-stream blocks to apply attention binding |
| `--masking_steps` | `all` | Denoising steps where hard masking is applied; or a comma-separated list |
| `--relaxed_timesteps` | `soft` | Behaviour at non-masked steps (`soft` keeps soft constraints, `full` removes them) |
| `--attention_setting` | `apitasmkernelnonlap` | `apitasmkernelnonlap` (default) or `apitasmkernelnonlapstrict` or `full` (no masking) |
| `--num_samples` | `None` | Limit to N samples (useful for debugging) |

## Attention Settings

- `apitasmkernelnonlap` — MICE attention processor with kernel-based spatial decay
- `apitasmkernelnonlapstrict` — same, with stricter cross-instance isolation
- `full` — unmodified baseline attention (no spatial binding)
