import numpy as np
import tensorflow as tf
import os

from tokenizer import load_tokenizer

def create_tf_dataset(corpus: str, tokenizer_path: str, max_len: int, batch_size: int):
    """
    Prepares training and validation tf.data.Dataset objects.

    Args:
        corpus (str): The cleaned text corpus.
        tokenizer_path (str): Path to the trained tokenizer file.
        max_len (int): The length of each sequence.
        batch_size (int): The batch size for the dataset.

    Returns:
        A tuple of (train_dataset, validation_dataset, vocab_size).
    """
    print("\nStarting Data Preparation for TF Dataset Pipeline...")
    
    tokenizer = load_tokenizer(tokenizer_path)
    encoded_text = tokenizer.encode(corpus).ids
    vocab_size = tokenizer.get_vocab_size()

    sequences = []
    for i in range(len(encoded_text) - max_len):
        sequences.append(encoded_text[i: i + max_len])
    
    sequences = np.array(sequences)
    X = sequences[:, :-1]
    y = sequences[:, -1]
    print(f"Created {len(X)} sequences.")

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    val_size = len(X) // 5  # 20% split for validation
    train_size = len(X) - val_size
    
    # Shuffling before splitting to ensure randomness
    dataset = dataset.shuffle(buffer_size=10000) 
    
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)
    
    # Cache the dataset in memory
    # Create batches and prefetch for performance
    # Autotune the prefetching to figure out the best number of batches to prefetch
    train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print("\nTF Dataset pipeline created and optimized.\n")
    print(f"Training dataset size: {train_size} samples")
    print(f"Validation dataset size: {val_size} samples")

    return train_dataset, validation_dataset, vocab_size