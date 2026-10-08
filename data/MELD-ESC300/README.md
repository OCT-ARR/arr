# 📚 MELD-ESC300

The included `test.jsonl.gz` is the 300-example, seed-42 external transfer
split described in the manuscript. It is used only for evaluation.

Selection uses at least two context turns, target responses of 4–40 words,
non-negative target sentiment, target emotions in `neutral`, `joy`, or
`surprise`, a speaker change, and an available preceding-utterance video.
There are 208 strict cases (immediately preceding negative affect) and 92
relaxed cases (negative affect within the three most recent preceding turns).
Each input uses at most six context turns and five uniformly sampled frames
from the immediately preceding utterance, excluding target-response video.

Run `python tools/unpack_data.py` from the repository root. Rebuild the JSONL
and frames with [build_meld_esc300.py](../../src/generalization_data_synthesis/build_meld_esc300.py),
as documented in [the reconstruction guide](../../src/generalization_data_synthesis/README.md).
Frames remain local under `data/MELD-ESC300/frames/`.

This controlled external transfer set derives from scripted TV dialogue;
all 300 examples are evaluated. To collect and aggregate expert validity
annotations, follow [human_preference](../../experiments/human_preference/README.md).

[Repository overview](../../README.md)
