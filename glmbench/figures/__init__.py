"""Every plot in the study.

    style.py          shared matplotlib defaults, palette, saving figures and tables
    significance.py   the significance marks drawn on the comparison figures
    main.py           Fig. 1, 2, 4-7 and the k-mer sweep (Fig. S1)
    supplementary.py  every layer of every metric, scaling and distance (Fig. 8)
    statistics.py     the significance plots

Figure numbers are the repository's; the article's numbering is in the README.  All
of them have the same two-stage shape: a build step turns the metric tables into a
small CSV under ``results/figure_data/``, and a draw step renders that CSV.
Restyling a figure therefore never re-reads a metric table.
"""

from . import main, statistics, style, supplementary  # noqa: F401
