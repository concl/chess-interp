"""Model, tokenizer, and adapter loading helpers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from gpt_chess.config import ModelConfig
from gpt_chess.tokenization import DirectTokenMapper, load_tokenizer

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def resize_embeddings_if_needed(model: Any, tokenizer: Any) -> None:
    """Match model embeddings to tokenizer size after adding chess tokens."""

    current_size = model.get_input_embeddings().num_embeddings
    if current_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))


def load_base_model_and_tokenizer(
    config: ModelConfig,
) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase", DirectTokenMapper]:
    """Load the configured base model and its direct chess token mapper."""

    import torch
    from transformers import AutoModelForCausalLM

    tokenizer, _ = load_tokenizer(config.model_id)

    model_kwargs: dict[str, object] = {}
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    resize_embeddings_if_needed(model, tokenizer)

    model.config.pad_token_id = tokenizer.pad_token_id
    mapper = DirectTokenMapper.from_tokenizer(tokenizer)
    return model, tokenizer, mapper


def attach_lora_adapter(model: Any, config: ModelConfig) -> Any:
    """Attach a LoRA adapter unless the model already has one."""

    if not config.use_lora or hasattr(model, "peft_config"):
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise ImportError(
            "LoRA training requires `peft`. Install it or set use_lora=False."
        ) from error

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def load_tokenizer_for_adapter(
    base_model_id: str,
    adapter_dir: str | None,
) -> tuple["PreTrainedTokenizerBase", int]:
    """Prefer a tokenizer saved with an adapter, falling back to the base model."""

    from transformers import AutoTokenizer

    if adapter_dir is not None and os.path.isdir(adapter_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        except OSError:
            return load_tokenizer(base_model_id)
    else:
        return load_tokenizer(base_model_id)

    added_tokens = tokenizer.add_special_tokens(
        {
            "additional_special_tokens": ["<CHESS>", "</CHESS>"],
            "pad_token": "<PAD>",
        }
    )
    return tokenizer, added_tokens


def load_adapter_model(
    adapter_dir: str,
    *,
    fallback_model_id: str = "gpt2",
    device: str | "torch.device" | None = None,
) -> tuple["PreTrainedModel", "PreTrainedTokenizerBase", DirectTokenMapper]:
    """Load a saved PEFT adapter for legal-move scoring."""

    from transformers import AutoModelForCausalLM

    try:
        from peft import PeftConfig, PeftModel
    except ImportError as error:
        raise ImportError("Adapter inference requires `peft`.") from error

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_id = peft_config.base_model_name_or_path or fallback_model_id
    tokenizer, _ = load_tokenizer_for_adapter(base_model_id, adapter_dir)

    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    resize_embeddings_if_needed(model, tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, adapter_dir, ignore_mismatched_sizes = True)
    resize_embeddings_if_needed(model, tokenizer)
    if device is not None:
        model = model.to(device)
    model.eval()

    mapper = DirectTokenMapper.from_tokenizer(tokenizer)
    return model, tokenizer, mapper

