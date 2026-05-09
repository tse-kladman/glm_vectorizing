import re
import os
import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



def extract_accession_numbers(accession_string):
    """
    Извлекает все номера доступа из строки вида:
    "RNA1: AB009656; RNA2: AB010376; RNA3: AB010377; RNA4: AB010378; RNA5: AB000403; RNA6: AB000404"
    """
    if pd.isna(accession_string):
        return []
    
    # Используем регулярное выражение для поиска номеров доступа
    # (последовательности букв и цифр, обычно начинающиеся с 1-2 букв и затем цифр)
    pattern = r'[A-Z]{1,2}\d{5,7}'
    accessions = re.findall(pattern, str(accession_string))
    return accessions


def sample_per_class(df, n_samples=10, random_state=42):
    """Выбирает n_samples случайных последовательностей для каждого класса"""
    sampled_dfs = []
    
    for class_label in df['Class'].unique():
        class_df = df[df['Class'] == class_label]
        
        # Если в классе меньше n_samples, берем все
        n_to_sample = min(n_samples, len(class_df))
        
        # Случайный выбор
        sampled_class_df = class_df.sample(n=n_to_sample, random_state=random_state)
        sampled_dfs.append(sampled_class_df)
    
    # Объединяем все выборки
    return pd.concat(sampled_dfs, ignore_index=True)


def split_sequence_into_chunks(sequence, chunk_size=4096):
    """Разбивает последовательность на чанки указанного размера"""
    if not sequence:
        return ['']
    
    chunks = []
    for i in range(0, len(sequence), chunk_size):
        chunk = sequence[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# Функция для создания чанков с сохранением меток
def create_chunked_dataset(df, chunk_size=4096, max_chunks_per_sequence=None):
    """Создает новый датасет с разбитыми на чанки последовательностями"""
    chunked_data = []
    
    for _, row in df.iterrows():
        sequence = row['seq']
        label = row['label']
        
        # Разбиваем последовательность на чанки
        chunks = split_sequence_into_chunks(sequence, chunk_size)

        if max_chunks_per_sequence is not None:
            # Генерируем рандомные числа для среза
            start = np.random.randint(len(chunks))
            end = start + max_chunks_per_sequence
            
            # Ограничиваем количество чанков
            chunks = chunks[start:end]
        
        # Создаем отдельную запись для каждого чанка
        for chunk in chunks:
            chunked_data.append({
                'seq': chunk,
                'label': label,
                'original_seq_length': len(sequence),
                'chunk_length': len(chunk),
                'chunk_index': chunks.index(chunk),  # Добавляем индекс чанка
                'total_chunks': len(chunks)  # Добавляем общее количество чанков
            })
    
    return chunked_data



def save_to_fasta(df, output_file, include_original=True):
    """
    Сохраняет в fasta
    
    Parameters:
    df : pandas.DataFrame
        DataFrame с колонками 'class' и 'sequence'
    output_file : str
        Путь к выходному FASTA файлу
    include_original : bool
        Включать ли оригинальный класс в ID
    """
    records = []
    
    for idx, row in df.iterrows():
        original_class = str(row['label'])
        sequence = str(row['seq']).strip()
        sequence = sequence.replace(" ", "").replace("\n", "")
        
        # Создаем уникальный ID
        if include_original:
            unique_id = f"{original_class}_idx{idx}"
        else:
            unique_id = f"seq_{idx}"
        
        # Создаем запись
        seq_record = SeqRecord(
            Seq(sequence),
            id=unique_id,
            description=f"original_class={original_class}"
        )
        records.append(seq_record)
    
    SeqIO.write(records, output_file, "fasta")