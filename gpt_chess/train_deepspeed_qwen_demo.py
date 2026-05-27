"""Demo DeepSpeed fine-tuning entrypoint for Qwen chess-move training.

GGUF files are llama.cpp inference artifacts, so they are not the right input
format for Hugging Face + DeepSpeed training. Fine-tune a trainable
Transformers checkpoint such as ``Qwen/Qwen3.5-9B`` first, then export or
quantize the result to GGUF after training if local llama.cpp inference is the
goal.

Example single-node launch:

    deepspeed --num_gpus 4 -m gpt_chess.train_deepspeed_qwen_demo \
        --model-id Qwen/Qwen3.5-9B \
        --dataset-split "train[:500]" \
        --output-dir models/chess_qwen35_9b_lora

For a quick wiring check without downloading a 9B model:

    python -m gpt_chess.train_deepspeed_qwen_demo --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from gpt_chess.config import DataConfig, ModelConfig, TrainerConfig
from gpt_chess.data import tokenize_dataset
from gpt_chess.modeling import attach_lora_adapter, resize_embeddings_if_needed
from gpt_chess.tokenization import (
    CHESS_END,
    CHESS_START,
    PAD_TOKEN,
    DirectTokenMapper,
)


DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3.5-9B"
QWEN_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_qwen_model_and_tokenizer(
    config: ModelConfig,
    *,
    torch_dtype: str,
    trust_remote_code: bool,
):
    """Load official Qwen3.5 models"""

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    dtype_by_name = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_kwargs: dict[str, object] = {
        "torch_dtype": dtype_by_name[torch_dtype],
        "trust_remote_code": trust_remote_code,
    }
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map

    try:
        processor = AutoProcessor.from_pretrained(
            config.model_id,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=trust_remote_code,
        )

    special_tokens: dict[str, str | list[str]] = {
        "additional_special_tokens": [CHESS_START, CHESS_END],
    }
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = PAD_TOKEN
    tokenizer.add_special_tokens(special_tokens)

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_id,
            **model_kwargs,
        )
    except (OSError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)

    resize_embeddings_if_needed(model, tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    mapper = DirectTokenMapper.from_tokenizer(tokenizer)
    return model, tokenizer, mapper


def build_deepspeed_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return a compact ZeRO config for LoRA fine-tuning"""

    return {
        "bf16": {"enabled": args.bf16},
        "fp16": {"enabled": not args.bf16},
        "zero_optimization": {
            "stage": args.zero_stage,
            "offload_optimizer": {
                "device": args.offload_optimizer_device,
                "pin_memory": True,
            },
            "offload_param": {
                "device": args.offload_param_device,
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.max_grad_norm,
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "train_batch_size": "auto",
        "steps_per_print": args.logging_steps,
        "wall_clock_breakdown": False,
    }


def make_configs(args: argparse.Namespace) -> tuple[ModelConfig, DataConfig, TrainerConfig]:
    """Build project configs"""

    model = replace(
        ModelConfig(),
        model_id=args.model_id,
        output_dir=args.output_dir,
        device_map=None,
        use_lora=not args.full_finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
    )
    data = replace(
        DataConfig(),
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        include_fen_metadata=not args.no_fen_metadata,
        position_policy=args.position_policy,
    )
    trainer = replace(
        TrainerConfig(),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        optim=args.optim,
        report_to=args.report_to,
        save_strategy=args.save_strategy,
    )
    return model, data, trainer


def train(args: argparse.Namespace):
    """Run the demo DeepSpeed fine-tuning job."""

    model_config, data_config, trainer_config = make_configs(args)
    deepspeed_config = build_deepspeed_config(args)

    if args.dry_run:
        print("DeepSpeed Qwen chess fine-tuning demo")
        print(f"model_id: {model_config.model_id}")
        print(f"dataset: {data_config.dataset_name} [{data_config.dataset_split}]")
        print(f"output_dir: {model_config.output_dir}")
        print(f"use_lora: {model_config.use_lora}")
        print(f"lora_targets: {', '.join(model_config.lora_target_modules)}")
        print(f"zero_stage: {args.zero_stage}")
        print(f"tqdm: {not args.disable_tqdm}")
        print("dry_run: skipped model and dataset downloads")
        return None

    from datasets import load_dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    output_dir = Path(model_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, mapper = load_qwen_model_and_tokenizer(
        model_config,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model = attach_lora_adapter(model, model_config)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    raw_dataset = load_dataset(
        data_config.dataset_name,
        split=data_config.dataset_split,
    )
    train_dataset = tokenize_dataset(
        raw_dataset,
        mapper=mapper,
        config=data_config,
        show_progress=not args.disable_tqdm,
    )
    print(f"Training examples extracted: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir=model_config.output_dir,
        per_device_train_batch_size=trainer_config.per_device_train_batch_size,
        gradient_accumulation_steps=trainer_config.gradient_accumulation_steps,
        learning_rate=trainer_config.learning_rate,
        num_train_epochs=trainer_config.num_train_epochs,
        logging_steps=trainer_config.logging_steps,
        optim=trainer_config.optim,
        report_to=trainer_config.report_to,
        save_strategy=trainer_config.save_strategy,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        fp16=not args.bf16,
        deepspeed=deepspeed_config,
        disable_tqdm=args.disable_tqdm,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    train_result = trainer.train()
    trainer.model.save_pretrained(model_config.output_dir)
    tokenizer.save_pretrained(model_config.output_dir)
    return train_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--output-dir", default="models/chess_qwen35_9b_lora")
    parser.add_argument("--dataset-name", default="patrickfrank1/chess-pgn-games")
    parser.add_argument("--dataset-split", default="train[:100]")
    parser.add_argument(
        "--position-policy",
        choices=["all_plies", "final_ply"],
        default="all_plies",
    )
    parser.add_argument(
        "--no-fen-metadata",
        action="store_true",
        help="Use only the 71-token expanded board string inside chess tags.",
    )
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Fine-tune all model weights instead of attaching a LoRA adapter.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(QWEN_LORA_TARGET_MODULES),
        help="Qwen projection modules to adapt with LoRA.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--save-strategy", default="epoch")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers loaders if needed.",
    )
    parser.add_argument("--zero-stage", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument(
        "--offload-optimizer-device",
        choices=["none", "cpu", "nvme"],
        default="none",
    )
    parser.add_argument(
        "--offload-param-device",
        choices=["none", "cpu", "nvme"],
        default="none",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bf16 mixed precision. Use --no-bf16 for fp16.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved demo config without loading data or model weights.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bars for dataset tokenization and training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
