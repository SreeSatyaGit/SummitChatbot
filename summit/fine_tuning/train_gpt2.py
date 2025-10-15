#!/usr/bin/env python3
"""Train / fine-tune a GPT-2 style causal LM using the Hugging Face Trainer.

Usage example:
  python3 fine_tuning/train_gpt2.py \
    --train-file data/train.txt \
    --output-dir models/gpt2-finetuned \
    --model-name-or-path gpt2 \
    --per-device-train-batch-size 4 \
    --num-train-epochs 3

The script accepts a plain text file (one example per line) or a CSV/JSON with a `text` column.
It tokenizes, groups into blocks and runs Trainer.
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from dataclasses import dataclass
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, List, Dict, Any
# Optional PEFT imports (import only when needed)
try:
    from peft import get_peft_model, LoraConfig, TaskType
except Exception:
    get_peft_model = None
    LoraConfig = None
    TaskType = None

@dataclass
class DataCollatorForCausalMasking:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # pad input_ids/attention_mask/labels to same length
        batch = self.tokenizer.pad(
            features, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )
        # ensure loss only on assistant tokens; ignore padding
        if "labels" in batch:
            labels = batch["labels"].clone()
            labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels
        return batch

def tokenize_dataset(dataset: Dataset, tokenizer, block_size: int) -> Dataset:
    eos_id = tokenizer.eos_token_id
    assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")

    def tok(example):
        text = example["text"]
        enc = tokenizer(
            text,
            truncation=True,
            max_length=block_size,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        ids = enc["input_ids"]
        attn = enc["attention_mask"]

        # ensure EOS at end
        if ids and ids[-1] != eos_id:
            ids = ids + [eos_id]
            attn = attn + [1]

        # labels = input_ids, but mask everything before (and incl.) <|assistant|>
        labels = list(ids)
        try:
            a_pos = ids.index(assistant_id)
        except ValueError:
            a_pos = None
        if a_pos is not None:
            for i in range(0, a_pos + 1):
                labels[i] = -100
        else:
            # if malformed example, ignore whole loss rather than train on user
            labels = [-100] * len(labels)

        return {"input_ids": ids, "attention_mask": attn, "labels": labels}

    tokenized = dataset.map(tok, remove_columns=dataset.column_names)
    return tokenized



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", required=True, help="Path to training data (txt/csv/json). For txt each line is one example")
    p.add_argument("--output-dir", required=True, help="Where to save the finetuned model")
    p.add_argument("--model-name-or-path", default="bigscience/bloom-560m", help="Pretrained model to start from (HF id or local path)")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--use-peft", action="store_true", help="Use PEFT/LoRA adapters instead of full fine-tune")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank r")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora-target-modules", type=str, default="", help="Comma-separated list of module names to apply LoRA to (optional)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_text_dataset(train_file: str):
    path = Path(train_file)
    ext = path.suffix.lower()
    if ext == ".txt":
        ds = load_dataset("text", data_files={"train": str(path)})
        # datasets 'text' produces {'text': ...}
        ds = ds["train"].filter(lambda x: x.get("text") is not None)
        # normalize to column 'text'
        return ds.map(lambda x: {"text": x["text"]})
    elif ext == ".csv":
        ds = load_dataset("csv", data_files={"train": str(path)})
        ds = ds["train"]
        if "text" not in ds.column_names:
            first_col = ds.column_names[0]
            ds = ds.rename_column(first_col, "text")
        return ds.map(lambda x: {"text": x["text"]})
    elif ext == ".json":
        ds = load_dataset("json", data_files={"train": str(path)})
        ds = ds["train"]
        if "text" not in ds.column_names:
            first_col = ds.column_names[0]
            ds = ds.rename_column(first_col, "text")
        return ds.map(lambda x: {"text": x["text"]})
    elif ext == ".jsonl":
        # jsonlines: parse manually into a Dataset
        import json as _json
        from datasets import Dataset as _Dataset

        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = _json.loads(line)
                if isinstance(obj, dict):
                    # prefer 'text' field, otherwise first value
                    text_val = obj.get("text") if obj.get("text") is not None else next(iter(obj.values()))
                else:
                    text_val = str(obj)
                records.append({"text": text_val})

        return _Dataset.from_list(records)
    else:
        raise ValueError("Unsupported train-file extension: use .txt, .csv, .json")




def tokenize_and_group(dataset, tokenizer, block_size: int = 512):
    # Tokenize the texts
    def tokenize_fn(examples):
        # don't return attention_mask here to avoid mismatched column lengths when grouping
        return tokenizer(
            examples["text"],
            return_special_tokens_mask=False,
            return_attention_mask=False,
            truncation=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # remove any unexpected mask columns to ensure a consistent schema
    extra_cols = [c for c in tokenized.column_names if c not in ("input_ids",)]
    if extra_cols:
        tokenized = tokenized.remove_columns(extra_cols)

    # concatenate and split into blocks
    def group_texts(examples):
        # concatenate lists of input_ids from the batch
        concatenated = sum(examples["input_ids"], [])
        total_length = len(concatenated)
        if total_length >= block_size:
            # drop the small remainder for now
            total_length = (total_length // block_size) * block_size
        else:
            total_length = 0

        result = {
            "input_ids": [
                concatenated[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        }
        # labels are the same as input_ids for causal LM
        result["labels"] = [list(r) for r in result["input_ids"]]
        return result

    grouped = tokenized.map(group_texts, batched=True)
    return grouped


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_text_dataset(args.train_file)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # GPT2 doesn't have pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Optionally wrap with PEFT/LoRA to reduce trainable params and memory footprint
    if args.use_peft:
        if get_peft_model is None:
            raise RuntimeError("PEFT requested but 'peft' is not installed. Install with: pip install peft")
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        if not target_modules:
            # common default modules for many causal LM architectures
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    print("Tokenizing and grouping...")
    train_dataset = tokenize_and_group(dataset, tokenizer, block_size=args.max_seq_length)

    data_collator = DataCollatorForCausalMasking(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=3,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        prediction_loss_only=True,
        fp16=args.fp16,
        seed=args.seed,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model and tokenizer to", args.output_dir)
    if args.use_peft:
        # PEFT models provide save_pretrained to persist adapter weights
        try:
            model.save_pretrained(args.output_dir)
        except Exception:
            # fallback to Trainer save
            trainer.save_model(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
