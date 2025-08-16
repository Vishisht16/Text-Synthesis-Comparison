from dataset_loader import load_and_clean_data
from tokenizer import create_and_train_bpe_tokenizer
from data_preparation import create_tf_dataset
import os

def run_data_pipeline(config: dict):
    """Executes the data pipeline using tf.data."""
    print("\nSTARTING DATA PIPELINE EXECUTION...")
    corpus = load_and_clean_data(config['data_file'], config['sample_size'])
    
    if not os.path.exists(config['tokenizer_path']):
        create_and_train_bpe_tokenizer(corpus, config['vocab_size'], config['tokenizer_path'])
    
    train_ds, val_ds, vocab_size = create_tf_dataset(
        corpus, config['tokenizer_path'], config['max_len'], config['batch_size']
    )
    
    print("\nDATA PIPELINE COMPLETE.")
    return train_ds, val_ds, vocab_size, config['max_len']
