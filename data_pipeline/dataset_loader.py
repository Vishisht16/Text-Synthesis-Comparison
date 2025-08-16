import pandas as pd
import string
import re

def load_and_clean_data(file_path: str, sample_size: int = 200000) -> str:
    """
    Loads news headlines from a CSV, cleans them, and returns a single text corpus.

    Args:
        file_path (str): The path to the CSV file.
        sample_size (int): The number of headlines to use from the dataset.

    Returns:
        str: A single string containing the cleaned and concatenated headlines.
    """
    print("\nStarting Data Loading and Cleaning...")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}. Original shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        print("Please make sure the dataset is available at the specified path.")
        return ""

    # Use the 'headline_text' column and handle potential missing values
    headlines = df['headline_text'].dropna()
    print(f"Loaded {len(headlines)} headlines.")

    # Create a single corpus from a subset of the data
    print(f"Using a sample of {sample_size} headlines for the corpus.")
    corpus = ' '.join(headlines[:sample_size])

    # Begin cleaning the corpus
    # 1. Convert to lowercase
    corpus = corpus.lower()

    # 2. Remove punctuation
    corpus = re.sub(f'[{re.escape(string.punctuation)}]', '', corpus)

    # 3. Remove numbers
    corpus = re.sub(r'\d+', '', corpus)
    
    # 4. Remove extra whitespace
    corpus = re.sub(r'\s+', ' ', corpus).strip()
    
    print("\nDataCleaning Complete.")
    print("Sample of the cleaned corpus:")
    print(corpus[:500])
    
    return corpus

if __name__ == '__main__':
    FILE_PATH = 'abcnews-date-text.csv'
    cleaned_corpus = load_and_clean_data(FILE_PATH, sample_size=50000)
    print(f"\nSuccessfully created a corpus with {len(cleaned_corpus.split())} words.")