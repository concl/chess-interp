import os
import sys
import argparse
import chess
import subprocess
import tkinter as tk
from tkinter import messagebox
import threading
import queue

# Unicode chess pieces
# White pieces are often drawn empty/outline in standard fonts if we just print the white character,
# but using standard Black unicode character and specifying fill colour might work better across OS.
# For simplicity, we just use the built-in unicode points.
PIECES = {
    chess.PAWN: ("♙", "♟"),
    chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"),
    chess.ROOK: ("♖", "♜"),
    chess.QUEEN: ("♕", "♛"),
    chess.KING: ("♔", "♚"),
}

class ChessGUI:
    def __init__(self, master, engine_process, board, player_is_white):
        self.master = master
        self.engine_process = engine_process
        self.board = board
        self.player_is_white = player_is_white
        self.square_size = 64
        
        self.canvas = tk.Canvas(master, width=8*self.square_size, height=8*self.square_size)
        self.canvas.pack()
        
        self.selected_square = None
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.queue = queue.Queue()
        self.master.after(100, self.process_queue)
        
        self.draw_board()
        
        if not self.player_is_white:
            self.engine_move()
            
    def draw_board(self):
        self.canvas.delete("all")
        for rank in range(8):
            for file in range(8):
                color = "#eeeed2" if (rank + file) % 2 != 0 else "#769656" # classic chess.com colors
                
                # If player is white, rank 0 is bottom (which is visually y = 7)
                v_rank = 7 - rank if self.player_is_white else rank
                v_file = file if self.player_is_white else 7 - file
                
                x1 = v_file * self.square_size
                y1 = v_rank * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags="square", outline="")
                
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                
                if self.selected_square == square:
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="#f6f669", width=4)
                
                if piece:
                    text_char = PIECES[piece.piece_type][0] if piece.color == chess.WHITE else PIECES[piece.piece_type][1]
                    # We render text as black for all unicode chars, to use the natural unicode chars
                    self.canvas.create_text(x1 + self.square_size/2, y1 + self.square_size/2, text=text_char, font=("Arial", 36), fill="black")

    def on_click(self, event):
        if not ((self.board.turn == chess.WHITE) == self.player_is_white):
            return # Engine's turn
            
        v_file = event.x // self.square_size
        v_rank = event.y // self.square_size
        
        # Determine internal rank/file based on view orientation
        file = v_file if self.player_is_white else 7 - v_file
        rank = 7 - v_rank if self.player_is_white else v_rank
        
        clicked_square = chess.square(file, rank)
        
        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.draw_board()
        else:
            move = chess.Move(self.selected_square, clicked_square)
            # handle naive queen promotion for simplicity
            piece_at_selected = self.board.piece_at(self.selected_square)
            if piece_at_selected and piece_at_selected.piece_type == chess.PAWN and \
               (chess.square_rank(clicked_square) == 0 or chess.square_rank(clicked_square) == 7):
                move = chess.Move(self.selected_square, clicked_square, promotion=chess.QUEEN)
                
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.draw_board()
                self.master.update()
                
                if not self.check_game_over():
                    self.engine_move()
            else:
                # If clicking own piece again, re-select
                piece = self.board.piece_at(clicked_square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = clicked_square
                else:
                    self.selected_square = None
                self.draw_board()

    def engine_move(self):
        if self.board.is_game_over():
            return
            
        def worker():
            self.engine_process.stdin.write(f"position fen {self.board.fen()}\n")
            self.engine_process.stdin.write("go depth 10\n")
            self.engine_process.stdin.flush()
            
            while True:
                line = self.engine_process.stdout.readline().strip()
                if line.startswith("bestmove"):
                    output = line.split()
                    if len(output) >= 2:
                        best_move = output[1]
                        self.queue.put(best_move)
                    break
                    
        threading.Thread(target=worker, daemon=True).start()
        
    def process_queue(self):
        try:
            best_move = self.queue.get_nowait()
            chess_move = chess.Move.from_uci(best_move)
            self.board.push(chess_move)
            self.selected_square = None # Reset player selection if any
            self.draw_board()
            self.check_game_over()
        except queue.Empty:
            pass
        self.master.after(100, self.process_queue)
        
    def check_game_over(self):
        if self.board.is_game_over():
            result = self.board.result()
            messagebox.showinfo("Game Over", f"Game Over! Result: {result}")
            return True
        return False

def main():
    parser = argparse.ArgumentParser(description="Play chess using a UCI engine with a GUI.")
    parser.add_argument(
        "engine_path",
        type=str,
        help="Path to the UCI engine program.",
    )
    parser.add_argument(
        "--starting_fen",
        type=str,
        help="FEN string for the starting position.",
    )
    parser.add_argument(
        "--engine_starts_first",
        action="store_true",
        help="If set, the engine will make the first move.",
    )
    
    args, unknown_args = parser.parse_known_args()
    
    if args.engine_path.endswith(".py"):
        interpreter_path = sys.executable
        args.engine_path = [interpreter_path, "-u", args.engine_path] + unknown_args
    else:
        args.engine_path = [args.engine_path] + unknown_args
    
    print(f"Starting UCI engine from: {args.engine_path}")
    engine_process = subprocess.Popen(
        args.engine_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    engine_process.stdin.write("uci\n")
    engine_process.stdin.flush()
    
    print("Waiting for engine to be ready...")
    while True:
        line = engine_process.stdout.readline().strip()
        if line == "uciok":
            break
        elif line != "":
            print(f"Engine output: {line}")
            
    board = chess.Board(args.starting_fen) if args.starting_fen else chess.Board()
    
    root = tk.Tk()
    root.title("Chess vs UCI Engine")
    root.resizable(False, False)
    
    player_is_white = not args.engine_starts_first
    app = ChessGUI(root, engine_process, board, player_is_white)
    
    print("GUI Started. Close the window to exit.")
    root.mainloop()
    
    print("Terminating engine...")
    engine_process.terminate()

if __name__ == "__main__":
    main()
