"""PGN parsing and supervised chess-move examples."""

from __future__ import annotations

import io
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import chess
import chess.pgn

from gpt_chess.config import DataConfig
from gpt_chess.tokenization import CHESS_END, CHESS_START, DirectTokenMapper

if TYPE_CHECKING:
    from datasets import Dataset


COMMENT_RE = re.compile(r"\{[^{}]*\}")
VARIATION_RE = re.compile(r"\([^()]*\)")
NAG_RE = re.compile(r"\$\d+")
RESULT_RE = re.compile(r"(1-0|0-1|1/2-1/2|\*)\s*$")
MOVE_RESULT_RE = re.compile(r"\b(1-0|0-1|1/2-1/2|\*)\s*$")


def clean_movetext(text: str) -> str:
    """Drop PGN headers and common annotations so python-chess can parse moves."""

    move_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("[")
    ]
    movetext = " ".join(move_lines)
    movetext = COMMENT_RE.sub(" ", movetext)
    movetext = VARIATION_RE.sub(" ", movetext)
    movetext = NAG_RE.sub(" ", movetext)
    movetext = RESULT_RE.sub("", movetext)
    return re.sub(r"\s+", " ", movetext).strip()


def read_game(text: str) -> chess.pgn.Game | None:
    """Parse one PGN game from dataset text."""

    movetext = clean_movetext(text)
    if not movetext.startswith("1."):
        return None
    return chess.pgn.read_game(io.StringIO(movetext))


def iter_game_texts(lines: Iterable[str]) -> Iterable[str]:
    """Reconstruct full PGN games from line-oriented dataset rows."""

    current_game: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip()
        if line or current_game:
            current_game.append(line)

        if MOVE_RESULT_RE.search(line):
            yield "\n".join(current_game)
            current_game = []


def expanded_board(board: chess.Board, *, expected_tokens: int = 71) -> str:
    """Return the board FEN with digits expanded to dots.

    The returned string keeps rank separators, so a normal chess board is
    exactly 64 squares plus 7 slashes, or 71 directly mapped tokens.
    """

    chars: list[str] = []
    for char in board.board_fen():
        if char.isdigit():
            chars.extend("." for _ in range(int(char)))
        else:
            chars.append(char)

    board_text = "".join(chars)
    if len(board_text) != expected_tokens:
        raise ValueError(
            f"Expanded board should be {expected_tokens} tokens, got {len(board_text)}: "
            f"{board_text}"
        )
    return board_text


def board_state_text(board: chess.Board, config: DataConfig) -> str:
    """Build the board state used between chess tags."""

    board_text = expanded_board(board, expected_tokens=config.board_token_count)
    if not config.include_fen_metadata:
        return board_text

    fen_parts = board.fen().split()
    return f"{board_text} {fen_parts[1]} {fen_parts[2]} {fen_parts[3]}"


def format_prompt(board: chess.Board, config: DataConfig) -> str:
    """Format one prompt for move prediction."""

    return f"{CHESS_START}\n{board_state_text(board, config)}\n{CHESS_END}\n"


def iter_game_examples(
    game: chess.pgn.Game,
    mapper: DirectTokenMapper,
    config: DataConfig,
) -> Iterable[dict[str, list[int]]]:
    """Yield one supervised example per pre-move board state by default."""

    board = game.board()
    moves = list(game.mainline_moves())

    for ply_index, move in enumerate(moves):
        use_position = (
            config.position_policy == "all_plies"
            or ply_index == len(moves) - 1
        )
        if use_position:
            prompt_ids = mapper.encode(format_prompt(board, config))
            completion_ids = mapper.encode(f"{move.uci()}\n")
            input_ids = prompt_ids + completion_ids

            yield {
                "input_ids": input_ids,
                "labels": [-100] * len(prompt_ids) + completion_ids,
                "attention_mask": [1] * len(input_ids),
            }

        board.push(move)


def extract_chess_examples(
    batch: dict[str, list[str]],
    *,
    mapper: DirectTokenMapper,
    config: DataConfig,
) -> dict[str, list[list[int]]]:
    """Dataset.map callback that expands each PGN game into training rows."""

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_mask: list[list[int]] = []

    for text in batch[config.text_column]:
        try:
            game = read_game(text)
            if game is None:
                continue

            for example in iter_game_examples(game, mapper, config):
                input_ids.append(example["input_ids"])
                labels.append(example["labels"])
                attention_mask.append(example["attention_mask"])
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
            continue

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def tokenize_dataset(
    dataset: "Dataset",
    *,
    mapper: DirectTokenMapper,
    config: DataConfig,
    show_progress: bool = False,
    progress_desc: str = "Tokenizing games",
) -> "Dataset":
    """Convert a raw PGN dataset into model-ready supervised examples."""

    from datasets import Dataset

    examples: list[dict[str, list[int]]] = []
    game_texts = iter_game_texts(row[config.text_column] for row in dataset)
    if show_progress:
        from tqdm.auto import tqdm

        game_texts = tqdm(game_texts, desc=progress_desc, unit="game")

    for game_text in game_texts:
        game = read_game(game_text)
        if game is None:
            continue
        examples.extend(iter_game_examples(game, mapper, config))

    return Dataset.from_list(examples)

