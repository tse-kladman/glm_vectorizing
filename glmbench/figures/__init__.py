"""Every plot in the study.

    style.py          shared matplotlib defaults, palette, saving figures and tables
    main.py           Fig. 1-7 of the article
    supplementary.py  every layer of every metric, grouped by metric
    statistics.py     the significance plots

All of them follow the same two-stage shape as before: a build step turns the
metric tables into a small CSV under ``results/figure_data/``, and a draw step
renders that CSV.  Restyling a figure therefore never re-reads a metric table.
"""

from . import main, statistics, style, supplementary  # noqa: F401
