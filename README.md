During this project we will be exploring mechanistic interpretability on board game models.

"Mechanistic Interpretability" (Mech Interp) is like doing neuroscience on AI. Instead of just looking at the final move an AI plays, we look at its internal "thoughts" (the high-dimensional vectors/activations, inside its hidden layers).

Our work will be roughly follow and build upon this: https://leela-interp.github.io/

Contents:

lc0_dist/

- Contains leela chess engine distribution for testing

leela_pytorch_impl/

- Hopefully we can work with the leela policy in pytorch

play_chess.py

- script to play against a given chess program that works with the UCI interface

## schedule

### Week 3: 

- [Colab solutions](https://colab.research.google.com/drive/1_dy32W_jAnJQdblnxsV9klIFlxdJOEan?usp=sharing)
- [Colab exercises](https://colab.research.google.com/drive/108rhOOeIU3d37gBVKnReDsaGI1LUIML9?usp=sharing)
- [Stanford NLP lecture: intro to probing](https://www.youtube.com/watch?v=ElDtkhqv5ZE)
- [Probing slides](https://docs.google.com/presentation/d/1vUKiuV3jKJ_Un9CJTO33IBpMd2dDdPSbVzH2qnil7cg/edit?usp=sharing)

- Introduction to Mech Interp
    - Like neuroscience for neural networks, we want to understand what goes on inside ML models - how and why they work.
- Linear probes
    - What are they? How do they work?
    - Practical implementation
- Working with game board models
    - Past research: Leela Chess Zero (see link above)
        - Which mech interp techniques applied and what did they show?
    - Looking ahead:
        - Try to research some board game models that have this kind of neural network architecture.
            - Since we have a smaller group, we’ll likely vote on a specific board game to focus on and explore different parts of the model using these mech interp techniques (starting with probes).
        - Othello and Chess are two big examples.

### Week 4:

Colab Links:
- [MLE notebook](https://colab.research.google.com/drive/1-dJ2DDmrdDL082NUPak9NpLMtmT7p0TK?usp=sharing)
    - [MLE solution](https://colab.research.google.com/drive/111DTxOYDZa3fYO7aXixVxb-zickqFpXz?usp=sharing)
- [LC0 notebook](https://colab.research.google.com/drive/14S6CqTzv9gKQultecLUm72EVPvlkLPii?usp=sharing)
    - [LC0 solution](https://colab.research.google.com/drive/1NIRcZjVStHYM0HUrxey0lOcbgrkaeYzX?usp=sharing)


Split into groups (MLE vs. AI Safety), Build out starter code 

- If we can port lc0 directly to PyTorch, we can use PyTorch as the backend.
- Split into groups
    - Give a few minutes for everyone to decide between groups
    - MLE (fine-tuning an existing GPT/chat model to play chess and output valid moves with a `<CHESS>` token)
    - AI Safety (working on reproducing parts of the lc0 paper, starting with linear probing on lc0 net)
- MLE group - intro to fine tuning
    - Goal (by end of meeting):
    - Add a special `<CHESS>` token to existing gpt model. Can be something like `gpt2` or a lightweight `qwen`.
    - Use huggingface dataset to fine-tune gpt model on PGN games (likely going to be from lichess).
    - Final version should show a working model where the input is `string = "<CHESS>\n1."` and the model continues by generating valid moves.
    - End goal: make a model that is strong/stable enough to inspect.
- AI Safety group - applying activation caching + probing to network
    - Goal (by end of meeting):
    - Load in (probably smaller version of) lc0 network
    - Use `TransformerLens`  to cache activations of an intermediate layer.
        - In future weeks may shift to `nnsight` instead?
    - Experiment with applying the torch hooks to adjust the activations (what happens when zeroing out activations at some layer, what do linear probes reveal, etc.)
    - Final version shows application of linear probes for some model understanding (e.g. “what square is the black king on?” or “is the white king in check?”)
    - End goal: interpret which signals in the model are doing useful work
- What next week will look like
    - MLE going to slightly modify how we use GPT → move from just sequence modeling (next token prediction) but add supervised QA
        - What should be implemented:
            - a dataset format that actually represents internal board state + answers questions (mixes move prediction and board QA?),
            - a model that still plays chess reasonably,
            - and a small eval script that checks both:
                - move quality
                - QA accuracy
    - AI Safety going to go from probing simple board features → running probes across multiple layers, comparing against random/control probes, pick strong features and attempt to analyze them causally (future weeks)
        - How this fits into LC0 paper logic: they used probes and patching to show that future-move information becomes decodable in later layers, and that specific heads support moving information between squares.
        - What kind of info should be gained:
            - “Feature X becomes decodable around layer Y.”
            - “Feature X is weak early and strongest later.”
            - “This suggests the model is building board-state representations internally.”

For ease of development (and access to free-tier GPUs) we’ll likely use Google Colab during the meeting. 

- If local/running on CPU is good enough, we can leave it in the Github.
    - Officers (before next meeting) will try to move all of the completed colab code into the Github - easier to debug and track changes.
- Otherwise, use the servers (see about getting project members server access if necessary).
