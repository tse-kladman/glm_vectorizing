"""ViraLM hidden states per encoder layer -- run it inside the ``viralm`` conda environment.

Adapted from the prediction script of ViraLM (https://github.com/ChengPENG-wolf/ViraLM):
the input is cut into ViraLM's own 1,000-nt windows and pushed through the fine-tuned
model, but instead of the virus score the hidden states of the requested encoder
layers are kept.  The pipeline calls it through ``conda run`` with this folder as the
working directory (``glmbench/representations/transformers.py``); the ViraLM model
files are expected in ``./model`` (``--model_pth``).

Output, for every requested layer L (1..12): ``<output_pth>/<FASTA stem>_layer_<L>.pkl``,
a pickled ``{"<record id>_<start>_<end>": vector}`` with one entry per window, in FASTA
order.  The vector is the mean of the layer's token states.  A 2,000-nt fragment gives
two windows; ``glmbench/representations/store.py`` averages them into one vector per
fragment.

``--batch_size`` must be 1: the encoder works on unpadded tokens, so the hidden state of
a batch is one (tokens, width) array, and its mean would mix the windows of the batch.
"""

import argparse
import csv
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import tqdm
import transformers
from Bio import SeqIO
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WINDOW = 1000          # ViraLM's window length
MIN_TAIL = 500         # a shorter tail of a sequence is dropped


def parse_args():
    parser = argparse.ArgumentParser(description="ViraLM hidden states per encoder layer")
    parser.add_argument("--input_pth", type=str, required=True, help="input FASTA")
    parser.add_argument("--output_pth", type=str, default="./TMP/",
                        help="output directory")
    parser.add_argument("--model_pth", type=str, default="./model",
                        help="directory of the ViraLM model files")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="must be 1 (see the module docstring)")
    parser.add_argument("--len_threshold", type=int, default=500,
                        help="sequences shorter than this are skipped")
    parser.add_argument("--cpu_threads", type=int, default=1,
                        help="data-loader workers (at most 4)")
    parser.add_argument("--filename", type=str, default="results",
                        help="name of the temporary window table")
    parser.add_argument("--layers_to_monitor", type=int, nargs="+",
                        default=list(range(1, 13)),
                        help="encoder layers to dump, numbered 1..12")
    return parser.parse_args()


def only_acgt(text, search=re.compile(r"[^ACGT]").search) -> bool:
    return not bool(search(text))


def write_windows(input_pth, table_path, len_threshold):
    """Cut every record into WINDOW-nt windows (plus a tail of at least MIN_TAIL nt);
    windows with a letter other than A/C/G/T are skipped."""
    with open(table_path, "w") as handle:
        handle.write("sequence,accession\n")
        for record in SeqIO.parse(input_pth, "fasta"):
            sequence = str(record.seq).upper()
            if len(sequence) < len_threshold:
                continue
            if len(sequence) >= WINDOW:
                last = 0
                for start in range(0, len(sequence) - WINDOW + 1, WINDOW):
                    window = sequence[start:start + WINDOW]
                    if only_acgt(window):
                        handle.write(f"{window},{record.id}_{start}_{start + WINDOW}\n")
                    last = start + WINDOW
                if len(sequence) - last >= MIN_TAIL:
                    window = sequence[last:]
                    if only_acgt(window):
                        handle.write(f"{window},{record.id}_{last}_{len(record.seq)}\n")
            elif only_acgt(sequence):
                handle.write(f"{sequence},{record.id}_0_{len(sequence)}\n")


@dataclass
class Collator:
    """Pads the token ids of a batch and keeps the window names as labels."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["accession"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return dict(input_ids=input_ids, labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


def load_model(model_pth, cache_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_pth, num_labels=2, trust_remote_code=True, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_pth, model_max_length=512, padding_side="right", truncation=False,
        use_fast=True, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on {device}.")
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    return model, tokenizer, device


def register_hooks(model, layers, states: dict) -> None:
    """Store the output of every requested encoder layer in ``states["layer_<L>"]``."""
    def hook_for(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            states[name] = output.detach().cpu()
        return hook

    bert = model.module.bert if hasattr(model, "module") else model.bert
    for layer in layers:
        bert.encoder.layer[layer - 1].register_forward_hook(hook_for(f"layer_{layer}"))


def make_loader(table_path, cache_dir, tokenizer, batch_size, cpu_threads):
    dataset = load_dataset("csv", data_files={"test": table_path}, cache_dir=cache_dir)
    tokenized = dataset.map(lambda batch: tokenizer(batch["sequence"], truncation=True),
                            batched=True, batch_size=512, remove_columns=["sequence"])
    return DataLoader(tokenized.with_format("torch")["test"], batch_size=batch_size,
                      collate_fn=Collator(tokenizer=tokenizer), pin_memory=True,
                      num_workers=min(cpu_threads, 4))


def collect(model, loader, device, layers, states: dict):
    """Per batch: the window names and the hidden state of every requested layer."""
    model.eval()
    names, per_layer = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            states.clear()
            names.append(batch.pop("labels"))
            model(**{key: value.to(device) for key, value in batch.items()})
            per_layer.append([states.get(f"layer_{layer}") for layer in layers])
    return names, per_layer


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise SystemExit("--batch_size must be 1: the mean over a batch mixes its windows")

    os.makedirs(args.output_pth, exist_ok=True)
    cache_dir = f"{args.output_pth}/cache"
    os.makedirs(cache_dir, exist_ok=True)
    table_path = f"{cache_dir}/{args.filename}_temp.csv"

    model, tokenizer, device = load_model(args.model_pth, cache_dir)
    write_windows(args.input_pth, table_path, args.len_threshold)
    with open(table_path) as handle:
        if len(list(csv.reader(handle))) < 2:
            raise SystemExit(f"no window of {args.input_pth} passed the filters")

    states = {}
    register_hooks(model, args.layers_to_monitor, states)
    loader = make_loader(table_path, cache_dir, tokenizer, args.batch_size, args.cpu_threads)
    names, per_layer = collect(model, loader, device, args.layers_to_monitor, states)

    stem = args.input_pth.split("/")[-1].split(".")[0]
    for index, layer in enumerate(args.layers_to_monitor):
        path = f"{args.output_pth}/{stem}_layer_{layer}.pkl"
        with open(path, "wb") as handle:
            pickle.dump({batch_names[0]: np.mean(np.array(hidden[index]), axis=0)
                         for batch_names, hidden in zip(names, per_layer)}, handle)
        print(path)
    os.remove(table_path)


if __name__ == "__main__":
    main()
