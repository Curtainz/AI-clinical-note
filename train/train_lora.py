"""LoRA / QLoRA SFT for Qwen3-4B-Instruct on the chat dataset from tools/build_sft_dataset.py.

This is the "模型训练" (model training) deliverable described in the repo
README. It is deliberately an **offline research track**:

* the competition package never ships model weights and the deployed model is
  called through an OpenAI-compatible endpoint, so the submitted run does not
  depend on this script;
* running it needs a CUDA GPU + PyTorch + PEFT (see train/README.md), none of
  which are assumed by ``main.py``.

The script imports heavy libraries lazily, so ``--dry-check`` can validate the
dataset and print the planned configuration on a machine without GPUs.

Usage (PowerShell, after pip install -r requirements-train.txt)::

    python train/train_lora.py --train-file data/sft/train.jsonl \
        --val-file data/sft/val.jsonl --output-dir checkpoints/qwen3-4b-lora

    python train/train_lora.py --dry-check --train-file data/sft/train.jsonl \
        --val-file data/sft/val.jsonl --output-dir checkpoints/qwen3-4b-lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _load_jsonl(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _check_dataset(path: Path, role: str, problems: list[str]) -> int:
    if not path.is_file():
        problems.append(f"{role} file not found: {path}")
        return 0
    samples = _load_jsonl(path)
    for index, sample in enumerate(samples[:2000]):
        messages = sample.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            problems.append(f"{role} sample #{index}: missing messages (need system/user/assistant)")
            continue
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles[:3] != ["system", "user", "assistant"]:
            problems.append(f"{role} sample #{index}: unexpected roles {roles}")
        assistant = messages[2].get("content") if isinstance(messages[2], dict) else None
        if not isinstance(assistant, str) or not assistant.strip():
            problems.append(f"{role} sample #{index}: empty assistant content")
    return len(samples)


def _apply_chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    """Apply the model chat template, keeping thinking traces disabled when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def _build_prompt_ids(tokenizer, messages: list[dict], max_seq_len: int) -> dict:
    """Tokenise a conversation, masking prompt tokens out of the loss."""
    prompt_text = _apply_chat_template(tokenizer, messages[:2], add_generation_prompt=True)
    full_text = prompt_text + messages[2]["content"] + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
        return_attention_mask=True,
    )
    input_ids = full["input_ids"]
    labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids) :]
    labels = labels[: len(input_ids)]
    return {"input_ids": input_ids, "attention_mask": full["attention_mask"], "labels": labels}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for the clinical-note task")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="base model id or local path")
    parser.add_argument("--train-file", type=Path, required=True, help="train.jsonl from build_sft_dataset.py")
    parser.add_argument("--val-file", type=Path, default=None, help="val.jsonl (optional)")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/qwen3-4b-lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1, help="per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="QLoRA 4-bit (default on)")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false", help="LoRA in bf16 instead of QLoRA")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--allow-cpu", action="store_true", help="allow training on CPU (very slow, smoke tests only)")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-check", action="store_true", help="validate data/config without importing torch")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems: list[str] = []
    train_count = _check_dataset(args.train_file, "train", problems)
    val_count = _check_dataset(args.val_file, "validation", problems) if args.val_file else 0

    config = {
        "base_model": args.model,
        "train_samples": train_count,
        "val_samples": val_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "quantization": "4bit-nf4 (QLoRA)" if args.load_in_4bit else "bf16 (LoRA)",
        "max_seq_len": args.max_seq_len,
        "gradient_checkpointing": args.gradient_checkpointing,
        "output_dir": str(args.output_dir),
    }

    if args.dry_check:
        print(json.dumps({"config": config, "problems": problems}, ensure_ascii=False, indent=2))
        if problems:
            print("DRY CHECK FAILED: dataset problems found", file=sys.stderr)
            return 1
        print("DRY CHECK OK (heavy dependencies are not imported in this mode)")
        return 0

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 2
    if not args.train_file.is_file():
        print(f"ERROR: train file not found: {args.train_file}", file=sys.stderr)
        return 2

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - depends on the local environment
        print(
            "ERROR: training dependencies are missing "
            f"({exc}).\nInstall them with: pip install -r requirements-train.txt\n"
            "Dataset preparation and --dry-check work without them.",
            file=sys.stderr,
        )
        return 3

    if not torch.cuda.is_available() and not args.allow_cpu:
        print(
            "ERROR: no CUDA device found. Qwen3-4B SFT is not practical on CPU; "
            "pass --allow-cpu only for smoke tests.",
            file=sys.stderr,
        )
        return 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict = {"trust_remote_code": True, "torch_dtype": compute_dtype}
    if args.load_in_4bit and device == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.load_in_4bit and device == "cuda":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.2f}%)")

    train_samples = _load_jsonl(args.train_file)
    encoded_train = [_build_prompt_ids(tokenizer, sample["messages"], args.max_seq_len) for sample in train_samples]
    encoded_val = (
        [_build_prompt_ids(tokenizer, sample["messages"], args.max_seq_len) for sample in _load_jsonl(args.val_file)]
        if args.val_file
        else []
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if encoded_val else "no",
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=(device == "cuda" and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if (args.load_in_4bit and device == "cuda") else "adamw_torch",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val or None,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"adapter saved to {args.output_dir}")
    print("next: python train/evaluate.py --adapter " + str(args.output_dir) + " --val-file " + str(args.val_file or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
