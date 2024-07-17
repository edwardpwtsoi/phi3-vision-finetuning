# Phi-3-Vision Finetuning

## Installation

```bash
pip install -r requirements.txt

# (optional) flash attention -- Ampere+ GPUs (e.g., A100, H100)
pip install ninja
MAX_JOBS=32 pip install flash-attn --no-build-isolation

# (optional) QLoRA -- Turing+ GPUs (e.g., RTX 8000)
pip install bitsandbytes==0.43.1

# evaluation
pip install table_recognition_metric
```

## Quick start

```bash
torchrun --nproc_per_node=8 finetune_hf_trainer.py --full_train --use_flash_attention --use_lora --bf16
```
