During this project we will be exploring mechanistic interpretability on board game models.

"Mechanistic Interpretability" (Mech Interp) is like doing neuroscience on AI. Instead of just looking at the final move an AI plays, we look at its internal "thoughts" (the high-dimensional vectors/activations, inside its hidden layers).

Contents:

lc0_dist/

- Contains leela chess engine distribution for testing

leela_pytorch_impl/

- Hopefully we can work with the leela policy in pytorch

play_chess.py

- script to play against a given chess program that works with the UCI interface