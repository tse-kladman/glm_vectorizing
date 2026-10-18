#!/usr/bin/env bash
# Reference benchmark (ARI, P@5, silhouette) of the two classical baselines that were
# added later -- the k-mer profile with Jensen-Shannon (kmer_js) and MinHash -- so that
# their rows of Table 2 / Fig. 2 come from committed tables instead of being read off
# a figure.  Run on the server, from the repository root, where data/random_samples/
# (the 50 reference samples used for every other model) already exists.
#
# CPU only, no GPU: extraction of k-mer profiles / sketches and the benchmark take
# minutes.  The word lengths are taken from results/metrics/selection.json (k = 4 for
# kmer_js, k = 6 for MinHash); nothing is re-tuned.
#
# Do NOT run the 'samples' stage here: it would redraw the reference samples, and the
# baselines would then be scored on other subsets than the language models.

set -euo pipefail

MODELS="kmer_js minhash"

git pull --ff-only

test -f data/random_samples/sample_0.fa || { echo "no reference samples in data/random_samples"; exit 1; }
python - <<'EOF'
import json
selection = json.load(open("results/metrics/selection.json"))
for key, k in (("kmer_js", 4), ("minhash", 6)):
    assert selection[key]["k"] == k, f"{key}: selection.json has k={selection[key]['k']}, expected {k}"
print("selection.json: kmer_js k=4, minhash k=6")
EOF

# 1. profiles / sketches of the 50 reference samples (skips files that already exist;
#    add --force if earlier vectors were made with another k or seed)
python run_pipeline.py extract --models $MODELS

# 2. ARI, P@5, silhouette (and P5_lgo where sample manifests exist) per reference sample
python run_pipeline.py benchmark --models $MODELS --force

# 3. check and publish the tables
for model in $MODELS; do
  for metric in ARI P5 silh; do
    test -f "results/metrics/$model/$metric.xlsx" || { echo "missing $model/$metric.xlsx"; exit 1; }
  done
done
python - <<'EOF'
import pandas as pd
for model in ("kmer_js", "minhash"):
    for metric in ("ARI", "P5", "silh"):
        table = pd.read_excel(f"results/metrics/{model}/{metric}.xlsx")
        print(f"{model:8} {metric:5} mean over {table.shape[1]} samples = {table.iloc[0].mean():.4f}")
EOF

git add results/metrics/kmer_js/*.xlsx results/metrics/minhash/*.xlsx results/run_manifest.json
git commit -m "Reference benchmark tables for the Jensen-Shannon and MinHash baselines"
git push
