"""Configuration for GPT chess fine-tuning experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


PositionPolicy = Literal["all_plies", "final_ply"]


@dataclass(frozen=True)
class ModelConfig:
    """Model and adapter settings.

    Change ``model_id`` to swap the base causal language model, for example
    ``"gpt2"`` or ``"Qwen/Qwen2.5-0.5B"``.
    """

    model_id: str = "gpt2"
    output_dir: str = "chess_model"
    device_map: str | None = "auto"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("c_attn", "c_proj")


@dataclass(frozen=True)
class DataConfig:
    """Dataset and board-token mapping settings."""

    dataset_name: str = "patrickfrank1/chess-pgn-games"
    dataset_split: str = "train[:500]"
    text_column: str = "text"
    board_token_count: int = 71
    include_fen_metadata: bool = True
    position_policy: PositionPolicy = "all_plies"


@dataclass(frozen=True)
class TrainerConfig:
    """Hugging Face trainer settings."""

    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    logging_steps: int = 10
    optim: str = "adamw_torch"
    report_to: str = "none"
    save_strategy: str = "no"
    deepspeed_config: str = "gpt_chess/ds_config.json"


@dataclass(frozen=True)
class GGUFConfig:
    """Config for local llama.cpp GGUF inference.

    The default points at Unsloth's Qwen3.5 9B Q4_0 GGUF. Swap ``repo_id`` and
    ``filename`` for any other Unsloth GGUF that fits in local RAM/VRAM.
    """

    repo_id: str = "unsloth/Qwen3.5-9B-GGUF"
    filename: str = "Qwen3.5-9B-Q4_0.gguf"
    local_dir: str = "models/gguf"
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    n_threads: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 16
    use_chat_template: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config used by scripts."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    gguf: GGUFConfig = field(default_factory=GGUFConfig)


DEFAULT_CONFIG = ExperimentConfig()

