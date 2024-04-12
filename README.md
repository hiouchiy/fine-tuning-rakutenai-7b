# LLM Fine-tuning Sample Code for RakutenAI-7b

This repository includes the following notebooks:
- rakutenai-7b_fine-tune_qlora
  - This notebook is used to fine-tune the 'RakutenAI-7b' model with QLoRA on a Japanese text dataset. It is primarily designed to run on a single GPU, such as the NVIDIA A100 (80GB).
- rakutenai-7b_fine-tune_qlora_multigpu
  - This notebook fine-tunes the 'RakutenAI-7b' model with QLoRA + DDP on a Japanese text dataset. It is primarily intended for use with multiple GPUs.
- rakutenai-7b_fine-tuning_deep-speed
  - This notebook fully fine-tunes the 'RakutenAI-7b' model on a Japanese text dataset. It is also targeted at setups using multiple GPUs.