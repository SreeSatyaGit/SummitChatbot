Fine-tuning GPT-2 (mini toolkit)

Files:
- train_gpt2.py - training script using Hugging Face Trainer

Quick start

1) Create a virtualenv and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install transformers datasets accelerate bitsandbytes -q
pip install torch  # or the correct torch wheel for your platform
```

2) Prepare data: a plain text file where each line is a training example (prompt+target or just target)

3) Run training

```bash
python fine_tuning/train_gpt2.py --train-file data/train.txt --output-dir fine_tuning/models/gpt2-finetuned --model-name-or-path gpt2 --per-device-train-batch-size 4 --num-train-epochs 3
```

Notes
- For small GPU (8GB) use small batch sizes and consider gradient accumulation.
- For large models, use `bitsandbytes` + `bnb` and `device_map='auto'` with `accelerate` configuration.
