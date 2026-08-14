# GPT Chess Models

This folder contains a small pipeline for fine-tuning causal language models to
predict UCI chess moves from board-state prompts, plus evaluation commands for
checking how well the model imitates held-out PGN moves.

## Setup

This project uses `uv` for Python environments and packages. Do not install
from `requirements.txt`; the dependencies live in `pyproject.toml` and
`uv.lock`.

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync
```

Run a lightweight smoke test:

```bash
uv run python -m gpt_chess.smoke_test
```

If the Unsloth GGUF repo asks for authentication, run:

```bash
uv run huggingface-cli login
```

## Train

Train the default GPT-2 LoRA adapter:

```bash
uv run python -m gpt_chess.train \
  --model-id gpt2 \
  --dataset-split "train[:500]" \
  --output-dir chess_model
```

Use deepspeed to train the model in parallel:

\(recommended\)
```
export TRITON_CACHE_DIR=/tmp/triton_cache
```

```bash
uv run deepspeed --module gpt_chess.train \
  --model-id gpt2 \
  --dataset-split "train[:500]" \
  --output-dir chess_model \
  --deepspeed-config gpt_chess/ds_config.json
```

Train a different Hugging Face causal LM by changing `--model-id`:

```bash
uv run python -m gpt_chess.train \
  --model-id Qwen/Qwen2.5-0.5B \
  --dataset-split "train[:500]" \
  --output-dir chess_model_qwen25
```

```bash
uv run deepspeed --module gpt_chess.train \
  --model-id Qwen/Qwen2.5-0.5B \
  --dataset-split "train[:500]" \
  --output-dir chess_model_qwen25
```

The training code uses every pre-move position in each game by default and maps
the expanded board to exactly 71 direct character tokens. To train only on the
last move of each game:

```bash
uv run python -m gpt_chess.train --position-policy final_ply
```

## Generate A Legal Game

Generate a game from a trained adapter by scoring every legal UCI move:

```bash
uv run python -m gpt_chess.play --adapter-dir chess_model --plies 20
```

## Evaluate Move Quality

The main metric is held-out next-move prediction: for each PGN position, the
script asks the model for the next UCI move and reports legal move rate, exact
match rate, and rank metrics when a backend can rank legal moves.
For raw GGUF generation, use `legal_rate` and `exact_match_rate`; rank metrics
are only meaningful for the adapter and random backends because they rank all
legal moves.

Random baseline:

```bash
uv run python -m gpt_chess.evaluate \
  --backend random \
  --dataset-split "train[500:600]" \
  --limit-positions 100
```

Fine-tuned adapter:

```bash
uv run python -m gpt_chess.evaluate \
  --backend adapter \
  --adapter-dir chess_model \
  --dataset-split "train[500:600]" \
  --limit-positions 100 \
  --output-json eval_results/gpt2_adapter.json
```

## Qwen3.5 9B GGUF Baseline

The default GGUF config uses Unsloth's Qwen3.5 9B Q4_0 quantization:

- repo: `unsloth/Qwen3.5-9B-GGUF`
- file: `Qwen3.5-9B-Q4_0.gguf`

Evaluate it as a raw prompting baseline:

```bash
uv run python -m gpt_chess.evaluate \
  --backend gguf \
  --gguf-repo-id unsloth/Qwen3.5-9B-GGUF \
  --gguf-filename Qwen3.5-9B-Q4_0.gguf \
  --dataset-split "train[500:520]" \
  --limit-positions 20 \
  --output-json eval_results/qwen35_9b_q4_0.json
```

On a one-position smoke test, this config loaded successfully and produced the
held-out PGN move:

```text
target_uci: e2e4
predicted_uci: e2e4
legal_rate: 1.0
exact_match_rate: 1.0
```

Generate a raw GGUF game:

```bash
uv run python -m gpt_chess.play_gguf \
  --gguf-repo-id unsloth/Qwen3.5-9B-GGUF \
  --gguf-filename Qwen3.5-9B-Q4_0.gguf \
  --plies 20
```

If you want the game to continue after an illegal model output, add
`--fallback-to-random`. That keeps the board legal but should not be counted as
raw model play strength.

This setup should work for any Unsloth quantized GGUF model that `llama.cpp`
can load, assuming the file fits in local RAM/VRAM and disk. Change
`--gguf-repo-id` and `--gguf-filename` to point at another GGUF. The generation
code first uses the chat template embedded in the GGUF metadata, then falls
back to a raw prompt if the model emits no text. It does not depend on
Qwen-specific tokenizer APIs because GGUF files include the tokenizer metadata.

Example with a smaller Unsloth model:

```bash
uv run python -m gpt_chess.evaluate \
  --backend gguf \
  --gguf-repo-id unsloth/Qwen3.5-0.8B-GGUF \
  --gguf-filename Qwen3.5-0.8B-Q4_0.gguf \
  --limit-positions 20
```

