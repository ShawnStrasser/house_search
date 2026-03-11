#!/usr/bin/env python3
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / 'tools/update_templates.py'), run_name="__main__")
