# 📚 MESC-FT

MESC-FT is the sole supervised and reinforcement-learning dataset in FACES.
The train, validation, and test JSONL archives contain 3,310, 368, and 450
examples, respectively. Training uses only the training split; checkpoint
selection uses the validation split.

Run `python tools/unpack_data.py` from the repository root to create the
`.jsonl` files. Targets use `<think>...</think><response>...</response>` with
flexible reasoning spans for Visual, Emotion, Situation, and Strategy.
Image paths are relative to `keyframes/` at the repository root.

The source MESC/MESC-FT datasets are credited in the manuscript; this release
contains their processed text supervision. Obtain raw visual assets under the
original source terms and reconstruct them with
[the visual preparation tools](../../src/image_data_synthesis/README.md).
The source videos and keyframes are not redistributed here.

[Repository overview](../../README.md)
