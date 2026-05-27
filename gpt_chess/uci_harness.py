"""UCI harness for adapter-trained GPT chess move models."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import TextIO

import chess

from gpt_chess.config import DataConfig
from gpt_chess.modeling import load_adapter_model
from gpt_chess.play import score_legal_uci_moves


DEFAULT_ADAPTER_DIR = "models/chess_qwen35_9b_lora"
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3.5-9B"


@dataclass(frozen=True)
class AdapterHarnessConfig:
    """Runtime settings for adapter-backed move selection."""

    adapter_dir: str = DEFAULT_ADAPTER_DIR
    fallback_model_id: str = DEFAULT_QWEN_MODEL_ID
    device: str | None = None
    include_fen_metadata: bool = True


class AdapterMoveHarness:
    """Load a trained adapter and choose legal UCI moves for board positions."""

    def __init__(self, config: AdapterHarnessConfig = AdapterHarnessConfig()) -> None:
        self.config = config
        self.data_config = DataConfig(include_fen_metadata=config.include_fen_metadata)
        self.model, self.tokenizer, self.mapper = load_adapter_model(
            config.adapter_dir,
            fallback_model_id=config.fallback_model_id,
            device=config.device,
        )

    def score_moves(self, board: chess.Board) -> list[tuple[float, str]]:
        """Return legal UCI moves sorted by model continuation log-probability."""

        return score_legal_uci_moves(
            self.model,
            self.mapper,
            board,
            self.data_config,
        )

    def choose_uci(self, board: chess.Board) -> str:
        """Return the model's highest-scoring legal UCI move."""

        scored_moves = self.score_moves(board)
        if not scored_moves:
            raise ValueError("Cannot choose a move from a terminal board position.")
        return scored_moves[0][1]

    def choose_move(self, board: chess.Board) -> chess.Move:
        """Return the model's highest-scoring legal move."""

        return chess.Move.from_uci(self.choose_uci(board))


class UCIAdapterEngine:
    """Minimal UCI protocol loop around an AdapterMoveHarness."""

    def __init__(
        self,
        harness: AdapterMoveHarness,
        *,
        name: str = "GPT Chess Adapter",
        out: TextIO = sys.stdout,
        err: TextIO = sys.stderr,
    ) -> None:
        self.harness = harness
        self.name = name
        self.out = out
        self.err = err
        self.board = chess.Board()

    def writeln(self, text: str) -> None:
        print(text, file=self.out, flush=True)

    def log(self, text: str) -> None:
        print(text, file=self.err, flush=True)

    def set_position(self, tokens: list[str]) -> None:
        if not tokens:
            return

        if "moves" in tokens:
            moves_index = tokens.index("moves")
            position_tokens = tokens[:moves_index]
            move_tokens = tokens[moves_index + 1 :]
        else:
            position_tokens = tokens
            move_tokens = []

        if position_tokens == ["startpos"]:
            board = chess.Board()
        elif position_tokens and position_tokens[0] == "fen":
            fen_tokens = position_tokens[1:]
            if not fen_tokens:
                raise ValueError("position fen requires a FEN string")
            board = chess.Board(" ".join(fen_tokens))
        else:
            raise ValueError(f"Unsupported position command: {' '.join(tokens)}")

        for move_uci in move_tokens:
            board.push(chess.Move.from_uci(move_uci))
        self.board = board

    def go(self) -> None:
        if self.board.is_game_over():
            self.writeln("bestmove 0000")
            return

        move_uci = self.harness.choose_uci(self.board)
        self.writeln(f"bestmove {move_uci}")

    def handle_line(self, line: str) -> bool:
        """Handle one UCI input line. Return False when the loop should exit."""

        stripped = line.strip()
        if not stripped:
            return True

        command, *tokens = stripped.split()
        try:
            if command == "uci":
                self.writeln(f"id name {self.name}")
                self.writeln("id author chess-interp")
                self.writeln("uciok")
            elif command == "isready":
                self.writeln("readyok")
            elif command == "ucinewgame":
                self.board.reset()
            elif command == "position":
                self.set_position(tokens)
            elif command == "go":
                self.go()
            elif command == "stop":
                return True
            elif command == "quit":
                return False
            else:
                self.log(f"info string ignored unsupported command: {stripped}")
        except Exception as error:
            self.log(f"info string error handling '{stripped}': {error}")
            if command == "go":
                self.writeln("bestmove 0000")
        return True

    def loop(self, inp: TextIO = sys.stdin) -> None:
        for line in inp:
            if not self.handle_line(line):
                break


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--fallback-model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--no-fen-metadata",
        action="store_true",
        help="Use only the 71-token board string inside chess tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    harness = AdapterMoveHarness(
        AdapterHarnessConfig(
            adapter_dir=args.adapter_dir,
            fallback_model_id=args.fallback_model_id,
            device=args.device,
            include_fen_metadata=not args.no_fen_metadata,
        )
    )
    UCIAdapterEngine(harness).loop()


if __name__ == "__main__":
    main()
