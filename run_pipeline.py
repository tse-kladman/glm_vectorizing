#!/usr/bin/env python
"""Entry point for the whole study.

    python run_pipeline.py --help            the stages
    python run_pipeline.py <stage> --help    the options of one stage

The work lives in the ``glmbench`` package; this file only forwards to it, so that
the repository has exactly one command to remember.
"""

from glmbench.cli import main

if __name__ == "__main__":
    main()
