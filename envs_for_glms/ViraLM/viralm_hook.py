from transformers import AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Dict, Sequence
from dataclasses import dataclass
from torch.nn import Softmax
from Bio import SeqIO
from torch import nn
import transformers
import numpy as np
import threading
import argparse
import torch
import csv
import re
import os
import shutil
import tqdm
import pickle 

def parse_args():
    parser = argparse.ArgumentParser(description='ViraLM sequence analysis parameters')
    
    parser.add_argument('--input_pth', type=str,
                       help='Input file path')
    parser.add_argument('--output_pth', type=str, default="./TMP/",
                       help='Output directory path')
    parser.add_argument('--model_pth', type=str, default="./model",
                       help='Model directory path')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--len_threshold', type=int, default=500,
                       help='Length threshold for sequences')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                       help='Score threshold for predictions')
    parser.add_argument('--cpu_threads', type=int, default=1,
                       help='Number of CPU threads to use')
    parser.add_argument('--filename', type=str, default="results",
                       help='Custom output filename')
    parser.add_argument('--layers_to_monitor', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       help='List of layers to monitor (space-separated integers)')
    
    return parser.parse_args()





def special_match(strg, search=re.compile(r'[^ACGT]').search):
    return not bool(search(strg))


def preprocee_data(input_pth, cache_dir, len_threshold, filename):
    # ViraLM's own 1,000-nt windows: a 2,000-nt fragment gives two entries.  They are
    # averaged back into one vector per fragment in glmbench/representations/store.py
    # (_viralm_groups); scoring the windows as separate points inflates ViraLM's metrics.
    frag_len = 1000
    
    # Create file in cache_dir
    temp_file_path = f"{cache_dir}/{filename}_temp.csv"
    with open(temp_file_path, "w") as f:
        f.write(f'sequence,accession\n')
        for record in SeqIO.parse(input_pth, "fasta"):
            global seq
            sequence = str(record.seq).upper()
            seq = sequence
            if len(sequence) < len_threshold:
                continue
            if len(sequence) >= frag_len:
                last_pos = 0
                for i in range(0, len(sequence) - frag_len + 1, 1000):
                    sequence1 = sequence[i:i + frag_len]
                    if special_match(sequence1):
                        f.write(f'{sequence1},{f"{record.id}_{i}_{i+frag_len}"}\n')
                    last_pos = i + frag_len
                if len(sequence) - last_pos >= 500:
                    sequence1 = sequence[last_pos:]
                    if special_match(sequence1):
                        f.write(f'{sequence1},{f"{record.id}_{last_pos}_{len(record.seq)}"}\n')
            elif len(sequence) >= len_threshold:
                if special_match(sequence):
                    f.write(f'{sequence},{f"{record.id}_{0}_{0+len(sequence)}"}\n')

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, str]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "accession"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = labels
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def tokenize_function(examples):
    return tokenizer(examples["sequence"], truncation=True)




def cpu_worker(batch, model, device):
    result = {}
    with torch.no_grad():
        labels = batch['labels']
        batch.pop('labels')
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits.cpu().numpy()
        #predictions = np.argmax(logits, axis=-1)

        for i in torch.arange(len(labels)):
            value = softmax(torch.tensor([logits[i][0], logits[i][1]])).tolist()
            segment_name = labels[i]
            seq_name = segment_name.rsplit('_', 2)[0]
            if seq_name not in result:
                result[seq_name] = []
            result[seq_name].append(value[1])
    return result




def get_hook(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            hidden_states_dict[layer_name] = output.detach().cpu()
        return hook

    
def setup_hooks(model, layer_indices=None):
    """Register hooks for specified layers"""
    hidden_states_dict = {}
    
    
    # Default to last layer if none specified
    if layer_indices is None:
        layer_indices = [12]  # Default to layer 11 (12th layer)
    
    # Handle both DataParallel and regular model
    bert_model = model.module.bert if hasattr(model, 'module') else model.bert
    
    # Register hooks for each specified layer
    for layer_idx in layer_indices:
        layer_name = f'layer_{layer_idx}'
        bert_model.encoder.layer[layer_idx-1].register_forward_hook(get_hook(layer_name))
    
    # Always hook embeddings for reference
    bert_model.embeddings.register_forward_hook(get_hook('embeddings'))
    
    return hidden_states_dict

def initialize_test_loader():
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Загружаем CSV как датасет
    test_dataset = load_dataset(
        'csv',
        data_files={'test': f'{cache_dir}/{filename}_temp.csv'},
        cache_dir=cache_dir
    )

    # Берём только первые 34 000 строк (если меньше — возьмёт все)
    test_dataset['test'] = test_dataset['test']#.select(range(34000, len(test_dataset['test'])))

    # Токенизация
    tokenized_datasets = test_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=512,
        remove_columns=["sequence"]
    )

    # Формат под PyTorch
    tokenized_datasets = tokenized_datasets.with_format("torch")

    # DataLoader с pin_memory и num_workers для ускорения
    test_loader = DataLoader(
        tokenized_datasets["test"],
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=min(cpu_threads, 4)
    )

    return test_loader


def initialize_model():
    global model, tokenizer, device

    model = AutoModelForSequenceClassification.from_pretrained(
        model_pth,
        num_labels=2,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_pth,
        model_max_length=512,
        padding_side="right",
        truncation=False,
        use_fast=True,
        trust_remote_code=True,
    )

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nRunning on {device}.')
    
    # Model setup
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs.')
        model = torch.nn.DataParallel(model)
    
    # Specify which layers you want to monitor (0-based or 1-based indices)
    # Example: [1, 6, 11] for layers 1, 6, and 11 (12th layer)
    layers_to_monitor = list(range(1, 13))  # Change this list to monitor different layers
    hidden_states_dict = setup_hooks(model, layers_to_monitor)
    
    
    return model, tokenizer, device

# Predict

def make_prediction(model, test_loader):
    softmax = Softmax(dim=0)
    model.eval()
    result = {}
    outputses = []
    states = []
    labelses = []
    if torch.cuda.is_available():
        with torch.no_grad():
            for step, batch in tqdm.tqdm(enumerate(test_loader)):
                # Clear previous hidden states
                hidden_states_dict.clear()
                
                labels = batch['labels']
                labelses.append(labels)
                batch.pop('labels')
                batch = {k: v.to(device) for k, v in batch.items()}
    
                outputs = model(**batch)
                outputses.append(outputs)
                logits = outputs.logits.detach().cpu().numpy()
                
                # Print outputs for all monitored layers
                int_states = []
                for layer_idx in layers_to_monitor:
                    layer_name = f'layer_{layer_idx}'
                    int_states.append(hidden_states_dict.get(layer_name, None))
                states.append(int_states)
                
                # Your existing processing
                for i in torch.arange(len(labels)):
                    value = softmax(torch.tensor([logits[i][0], logits[i][1]])).tolist()
                    segment_name = labels[i]
                    seq_name = segment_name.rsplit('_', 2)[0]
                    if seq_name not in result:
                        result[seq_name] = []
                    result[seq_name].append(value[1])
    else:
        # For CPU version
        tasks = []
        tasks = [pool.submit(cpu_worker, batch, model, device) for batch in test_loader]
        pool.shutdown(wait=True)
        for task in as_completed(tasks):
            predictions = task.result()
            for seq_name in predictions:
                if seq_name not in result:
                    result[seq_name] = []
                result[seq_name].extend(predictions[seq_name])
    return states, labelses

args = parse_args()
input_pth = args.input_pth
output_pth = args.output_pth
model_pth = args.model_pth
batch_size = args.batch_size
len_threshold = args.len_threshold
score_threshold = args.score_threshold
cpu_threads = args.cpu_threads
filename = args.filename
layers_to_monitor = args.layers_to_monitor



seq = ''
# Create output directory if it doesn't exist
os.makedirs(output_pth, exist_ok=True)
cache_dir = f'{output_pth}/cache'
os.makedirs(cache_dir, exist_ok=True)

model, tokenizer, device = initialize_model()



temp_file_path = f"{cache_dir}/{filename}_temp.csv"
if not os.path.exists(temp_file_path):
    print(f"Error: File {temp_file_path} was not created by preprocee_data.")


preprocee_data(input_pth, cache_dir, len_threshold, filename)
test_loader = initialize_test_loader()
hidden_states_dict = setup_hooks(model, layer_indices=layers_to_monitor)
states, labelses = make_prediction(model, test_loader)


file_suffix = input_pth.split('/')[-1].split('.')[0]
for i, layer in enumerate(layers_to_monitor):
    with open(f'{output_pth}/{file_suffix}_layer_{layer}.pkl', 'wb') as f:
        pickle.dump({l[0]:np.mean(np.array(a[i]), axis=0) for l, a in zip(labelses, states)}, f)
        print(f'{output_pth}/{file_suffix}_layer_{layer}.pkl')
os.remove(temp_file_path)
