#!/usr/bin/env python3
"""Rasterize the two original PDF figures for GitHub README display."""
from pathlib import Path
import shutil
import subprocess


def main():
    renderer = shutil.which('pdftoppm')
    if renderer is None:
        raise SystemExit('Install Poppler (pdftoppm) to regenerate previews.')
    root = Path(__file__).resolve().parents[1] / 'figures'
    (root / 'previews').mkdir(exist_ok=True)
    for stem, size in [('FACES_MAIN5', 1800), ('appendix_ablation_radar', 1100)]:
        subprocess.run([renderer, '-f', '1', '-l', '1', '-singlefile', '-cropbox', '-scale-to', str(size),
                        '-png', str(root / f'{stem}.pdf'), str(root / 'previews' / stem)], check=True)
        print(f'Rendered figures/previews/{stem}.png')


if __name__ == '__main__':
    main()
