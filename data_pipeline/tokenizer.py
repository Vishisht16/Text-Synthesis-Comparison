from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

# Import the data loading function from our other file
from dataset_loader import load_and_clean_data

def create_and_train_bpe_tokenizer(corpus: str, vocab_size: int, save_path: str):
    """
    Creates, trains, and saves a BPE tokenizer.

    Args:
        corpus (str): The text corpus to train the tokenizer on.
        vocab_size (int): The desired size of the vocabulary.
        save_path (str): Path to save the trained tokenizer file (e.g., 'bpe-tokenizer.json').
    """
    print(f"--- Starting BPE Tokenizer Training with vocab size {vocab_size} ---")
    
    # 1. Initialize a new Tokenizer with the BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # 2. Set the pre-tokenizer (how to split text before tokenization)
    tokenizer.pre_tokenizer = Whitespace()
    
    # 3. Initialize a trainer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
    
    # 4. Train the tokenizer
    temp_file = "temp_corpus.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(corpus)

    tokenizer.train([temp_file], trainer)
    
    # 5. Save the tokenizer
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")

    # Clean up the temporary file
    os.remove(temp_file)
    

def load_tokenizer(path: str) -> Tokenizer:
    """Loads a pre-trained tokenizer from a file."""
    return Tokenizer.from_file(path)

if __name__ == '__main__':
    # 1. Load and clean the data using our dataset module
    DATA_FILE = 'abcnews-date-text.csv'
    corpus_text = load_and_clean_data(DATA_FILE, sample_size=200000)

    # 2. Define tokenizer parameters
    VOCAB_SIZE = 10000
    TOKENIZER_PATH = 'bpe-tokenizer.json'

    # 3. Train and save the tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        create_and_train_bpe_tokenizer(corpus_text, VOCAB_SIZE, TOKENIZER_PATH)
    else:
        print(f"Tokenizer already exists at {TOKENIZER_PATH}")

    # 4. Load the tokenizer and test it
    my_tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    print("\n--- Testing the Tokenizer ---")
    sample_text = "australian Broadcasting corporation starts a new show"
    encoded = my_tokenizer.encode(sample_text)
    
    print(f"Original: {sample_text}")
    print(f"Encoded (IDs): {encoded.ids}")
    print(f"Encoded (Tokens): {encoded.tokens}")
    
    decoded = my_tokenizer.decode(encoded.ids)
    print(f"Decoded: {decoded}")