import os
import sys
import argparse
import chess
import subprocess

parser = argparse.ArgumentParser(description="Play chess using a UCI engine.")
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

# Pass arbitrary extra arguments to the engine
# (Alternatively, parsing known args in main() accomplishes this)

def main():
    
    args, unknown_args = parser.parse_known_args()
    
    if args.engine_path.endswith(".py"):
        interpreter_path = sys.executable
        args.engine_path = [interpreter_path, "-u", args.engine_path] + unknown_args
    else:
        args.engine_path = [args.engine_path] + unknown_args
    
    print(f"Starting UCI engine from: {args.engine_path}")
    # Start the UCI engine process
    engine_process = subprocess.Popen(
        args.engine_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    
    # Send the UCI initialization commands
    engine_process.stdin.write("uci\n")
    engine_process.stdin.flush()
    
    # Wait for the engine to be ready 
    while True:
        line = engine_process.stdout.readline().strip()
        if line == "uciok":
            break
        elif line != "":
            print(f"Engine output during initialization: {line}")
    
    # Set up the chess board
    board = chess.Board(args.starting_fen) if args.starting_fen else chess.Board()
    print("\nPlaying against the engine...")
    
    curr_turn = "Engine" if args.engine_starts_first else "Player"
    
    while not board.is_game_over():
        position_fen = board.fen()
        print("\nCurrent position:")
        print(board)
        
        if curr_turn == "Player":
            move = input("Enter your move (in UCI format, e.g., e2e4): ")
            try:
                chess_move = chess.Move.from_uci(move)
                if chess_move in board.legal_moves:
                    board.push(chess_move)
                    print(f"Player moves: {move}")
                    curr_turn = "Engine"
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Invalid move format. Try again.")
        else:
            # Send the current position to the engine
            engine_process.stdin.write(f"position fen {board.fen()}\n")
            engine_process.stdin.write("go depth 10\n")
            engine_process.stdin.flush()
            
            # Read the engine's move
            while True:
                line = engine_process.stdout.readline().strip()
                if line.startswith("bestmove"):
                    output = line.split()
                    best_move = output[1]
                    chess_move = chess.Move.from_uci(best_move)
                    board.push(chess_move)
                    print(f"Engine moves: {best_move}")
                    curr_turn = "Player"
                    break

if __name__ == "__main__":
    main()
