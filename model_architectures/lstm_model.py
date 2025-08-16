from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from metrics import perplexity

def build_lstm_model(hp, vocab_size, max_len):
    """
    Builds a Sequential LSTM model for Keras Tuner.

    Args:
        hp: Keras Tuner HyperParameters object.
        vocab_size (int): The size of the vocabulary.
        max_len (int): The max sequence length for input.

    Returns:
        A compiled Keras model.
    """
    model = Sequential()
    
    # 1. Embedding Layer
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=hp.Int('embedding_dim', min_value=64, max_value=256, step=64),
    ))
    
    # 2. LSTM Layer
    model.add(LSTM(
        units=hp.Int('lstm_units', min_value=100, max_value=300, step=50),
        return_sequences=False
    ))
    
    # 3. Dropout Layer for Regularization
    model.add(Dropout(
        rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    ))

    # 4. Output Layer
    model.add(Dense(vocab_size, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', perplexity]
    )
    
    return model