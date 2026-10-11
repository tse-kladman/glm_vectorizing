# No licence has been chosen yet -- the repository owner must pick one

Until a `LICENSE` file exists, the code is "all rights reserved" by default and nobody
else may legally reuse it. This file is a placeholder, not a licence. Delete it and add a
`LICENSE` (and a `license:` field in `CITATION.cff`) once you have decided.

Common choices for research code (not a recommendation, and not legal advice):

| Licence | In one line |
|---------|-------------|
| MIT | Very short and permissive; keep the copyright notice. |
| Apache-2.0 | Permissive, plus an explicit patent grant and a NOTICE mechanism. |
| BSD-3-Clause | Permissive like MIT, plus a no-endorsement clause. |

Things the owner has to check (nothing below is asserted here -- verify each):

* **Model licences.** The repository does not ship any model weights, but it downloads
  or calls them. Check the current licence and terms of use of **Evo 2** (and of the
  weights you actually use), **ViraLM** (https://github.com/ChengPENG-wolf/ViraLM -- the
  hook that dumps its hidden states is not in this repository), **DNABERT-2** and
  **DNABERT-S** (HuggingFace model cards), including any restriction on redistributing
  derived embeddings.
* **Data licences.** The ICTV VMR spreadsheet in `data/ictv_data/` and the GenBank
  genomes the reference samples are drawn from have their own terms of use.
* **Your co-authors / institution** may have a say in the choice.
