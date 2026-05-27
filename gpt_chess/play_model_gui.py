"""Click-to-move GUI for playing against a trained GPT chess adapter."""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from tkinter import messagebox

import chess

from gpt_chess.uci_harness import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_QWEN_MODEL_ID,
    AdapterHarnessConfig,
    AdapterMoveHarness,
    default_device,
)


LIGHT_SQUARE = "#eeeed2"
DARK_SQUARE = "#769656"
SELECTED_SQUARE = "#f6f669"
LAST_MOVE_SQUARE = "#baca44"


class HumanVsModelGUI:
    """Small Tkinter board that uses AdapterMoveHarness for model replies."""

    def __init__(
        self,
        root: tk.Tk,
        harness: AdapterMoveHarness,
        *,
        board: chess.Board,
        human_color: chess.Color,
    ) -> None:
        self.root = root
        self.harness = harness
        self.board = board
        self.human_color = human_color
        self.square_size = 72
        self.selected_square: chess.Square | None = None
        self.last_move: chess.Move | None = None
        self.move_queue: queue.Queue[tuple[chess.Move | None, Exception | None]] = queue.Queue()
        self.model_thinking = False

        self.canvas = tk.Canvas(
            root,
            width=8 * self.square_size,
            height=8 * self.square_size,
        )
        self.canvas.pack()
        self.status = tk.Label(root, text="", anchor="w")
        self.status.pack(fill=tk.X)

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.after(100, self.process_model_queue)
        self.draw_board()
        self.update_status()

        if self.board.turn != self.human_color:
            self.request_model_move()

    def is_human_turn(self) -> bool:
        return self.board.turn == self.human_color and not self.model_thinking

    def square_to_xy(self, square: chess.Square) -> tuple[int, int]:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if self.human_color == chess.WHITE:
            return file, 7 - rank
        return 7 - file, rank

    def xy_to_square(self, x: int, y: int) -> chess.Square:
        file = x // self.square_size
        rank = y // self.square_size
        if self.human_color == chess.WHITE:
            return chess.square(file, 7 - rank)
        return chess.square(7 - file, rank)

    def draw_board(self) -> None:
        self.canvas.delete("all")
        highlighted = set()
        if self.last_move is not None:
            highlighted.update((self.last_move.from_square, self.last_move.to_square))

        for square in chess.SQUARES:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            view_file, view_rank = self.square_to_xy(square)
            x1 = view_file * self.square_size
            y1 = view_rank * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size

            color = LIGHT_SQUARE if (file + rank) % 2 else DARK_SQUARE
            if square in highlighted:
                color = LAST_MOVE_SQUARE
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

            if self.selected_square == square:
                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    outline=SELECTED_SQUARE,
                    width=4,
                )

            piece = self.board.piece_at(square)
            if piece is not None:
                text_color = "white" if piece.color == chess.WHITE else "black"
                self.canvas.create_text(
                    x1 + self.square_size / 2,
                    y1 + self.square_size / 2,
                    text=piece.symbol(),
                    font=("Arial", 36, "bold"),
                    fill=text_color,
                )

    def update_status(self, text: str | None = None) -> None:
        if text is not None:
            self.status.config(text=text)
            return
        if self.board.is_game_over():
            self.status.config(text=f"Game over: {self.board.result()}")
        elif self.model_thinking:
            self.status.config(text="Model thinking...")
        elif self.is_human_turn():
            self.status.config(text="Your move")
        else:
            self.status.config(text="Model to move")

    def on_click(self, event: tk.Event) -> None:
        if not self.is_human_turn():
            return

        clicked_square = self.xy_to_square(event.x, event.y)
        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece is not None and piece.color == self.human_color:
                self.selected_square = clicked_square
                self.draw_board()
            return

        move = self.build_human_move(self.selected_square, clicked_square)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.last_move = move
            self.selected_square = None
            self.draw_board()
            if not self.check_game_over():
                self.request_model_move()
            return

        piece = self.board.piece_at(clicked_square)
        self.selected_square = (
            clicked_square
            if piece is not None and piece.color == self.human_color
            else None
        )
        self.draw_board()

    def build_human_move(
        self,
        from_square: chess.Square,
        to_square: chess.Square,
    ) -> chess.Move:
        piece = self.board.piece_at(from_square)
        promotion = None
        if (
            piece is not None
            and piece.piece_type == chess.PAWN
            and chess.square_rank(to_square) in (0, 7)
        ):
            promotion = chess.QUEEN
        return chess.Move(from_square, to_square, promotion=promotion)

    def request_model_move(self) -> None:
        if self.board.is_game_over() or self.model_thinking:
            return
        self.model_thinking = True
        self.update_status()
        board_snapshot = self.board.copy(stack=False)

        def worker() -> None:
            try:
                move = self.harness.choose_move(board_snapshot)
                self.move_queue.put((move, None))
            except Exception as error:
                self.move_queue.put((None, error))

        threading.Thread(target=worker, daemon=True).start()

    def process_model_queue(self) -> None:
        try:
            move, error = self.move_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self.process_model_queue)
            return

        self.model_thinking = False
        if error is not None:
            self.update_status(f"Model error: {error}")
            messagebox.showerror("Model Error", str(error))
        elif move is not None and move in self.board.legal_moves:
            self.board.push(move)
            self.last_move = move
            self.selected_square = None
            self.draw_board()
            self.check_game_over()
        else:
            self.update_status("Model returned an illegal move.")
        self.update_status()
        self.root.after(100, self.process_model_queue)

    def check_game_over(self) -> bool:
        if self.board.is_game_over():
            result = self.board.result()
            messagebox.showinfo("Game Over", f"Game over: {result}")
            self.update_status()
            return True
        self.update_status()
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--fallback-model-id", default=DEFAULT_QWEN_MODEL_ID)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--starting-fen", default=None)
    parser.add_argument(
        "--model-plays-white",
        action="store_true",
        help="Let the trained model move first as White.",
    )
    parser.add_argument(
        "--no-fen-metadata",
        action="store_true",
        help="Use only the 71-token board string inside chess tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    board = chess.Board(args.starting_fen) if args.starting_fen else chess.Board()
    human_color = chess.BLACK if args.model_plays_white else chess.WHITE
    harness = AdapterMoveHarness(
        AdapterHarnessConfig(
            adapter_dir=args.adapter_dir,
            fallback_model_id=args.fallback_model_id,
            device=args.device,
            include_fen_metadata=not args.no_fen_metadata,
        )
    )

    root = tk.Tk()
    root.title("Chess vs GPT Adapter")
    root.resizable(False, False)
    HumanVsModelGUI(root, harness, board=board, human_color=human_color)
    root.mainloop()


if __name__ == "__main__":
    main()
