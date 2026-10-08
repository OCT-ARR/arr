#!/usr/bin/env python3
"""Run the shared six-metric pipeline for saved local or API model responses."""
import runpy
from pathlib import Path


if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'api_baselines/run_metrics.py'),
                   run_name='__main__')
