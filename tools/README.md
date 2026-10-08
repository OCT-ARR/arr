# 🧰 Data and figure utilities

Run commands from the repository root.

| Script | Purpose |
| --- | --- |
| [unpack_data.py](unpack_data.py) | Decompress the six JSONL and three CSV data files to the paths used by training and evaluation |
| [render_figure_previews.py](render_figure_previews.py) | Render the two PDF figures as PNG previews using Poppler |

```bash
python tools/unpack_data.py
```

To regenerate PDF previews, install Poppler and run:

```bash
python tools/render_figure_previews.py
```

[Repository overview](../README.md)
