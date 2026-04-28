
from model import Lc0Model
from leela_board import LeelaBoard
import torch
import os
import argparse

PATH = os.path.dirname(os.path.abspath(__file__))
LC0_MODEL_PATH = os.path.join(PATH, "lc0-original.onnx")
USE_PROBABILITIES = False

parser = argparse.ArgumentParser(description="Leela Chess Zero UCI Bot.")
parser.add_argument(
    "--model_path",
    type=str,
    default=LC0_MODEL_PATH,
    help="Path to the Leela Chess Zero ONNX model.",
)

parser.add_argument(
    "--use_probabilities",
    action="store_true",
    help="If set, the bot will select moves based on probabilities instead of always choosing the best move.",
)

def main():
    args = parser.parse_args()
    global LC0_MODEL_PATH, USE_PROBABILITIES
    LC0_MODEL_PATH = args.model_path
    USE_PROBABILITIES = args.use_probabilities
    
    print(f"Loading Leela Chess Zero model from: {LC0_MODEL_PATH}")
    print(f"Using probabilities for move selection: {USE_PROBABILITIES}")
    
    model = Lc0Model(onnx_model_path=LC0_MODEL_PATH)
    print("Leela Chess Zero UCI Bot is ready. Waiting for 'uci' command to start...")
    
    wait_uci = input()
    if wait_uci.strip() != "uci":
        print("Expected 'uci' command to start the bot.")
        return
    
    print("uciok")
    
    while True:
        line = input().strip()
        if line.startswith("position fen"):
            fen = line[len("position fen "):].strip()
            print(f"Received position: {fen}")
            
            leela_input = LeelaBoard.from_fen(fen)
            
            with torch.no_grad():
                policy, win_draw_loss, moves_left = model.play(leela_input, return_probs=False)
                policy_dict = model.policy_as_dict(leela_input, policy)
                
                if not USE_PROBABILITIES:
                    best_move = max(policy_dict, key=policy_dict.get)
                else:
                    all_moves = list(policy_dict.keys())
                    probabilities = torch.softmax(torch.tensor(list(policy_dict.values())), dim=0)
                    select = torch.multinomial(probabilities, num_samples=1).item()
                    best_move = all_moves[select]
            
            print(f"bestmove {best_move}")
            
        elif line == "quit":
            break
    
    
print("Starting Leela Chess Zero UCI Bot...")
if __name__ == "__main__":
    main()

