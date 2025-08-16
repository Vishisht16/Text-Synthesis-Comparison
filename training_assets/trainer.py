import tensorflow as tf
import time
import os
from models.lstm_model import build_lstm_model
from models.gru_model import build_gru_model
from models.transformer_model import build_transformer_model
from metrics import perplexity

def train_model(strategy, model_name, best_hps, train_ds, val_ds, vocab_size, max_len, epochs=20):
    print(f"\n--- Starting Final Training for {model_name.upper()} ---")

    with strategy.scope():
        if model_name == 'lstm': model_builder = build_lstm_model
        elif model_name == 'gru': model_builder = build_gru_model
        elif model_name == 'transformer': model_builder = build_transformer_model
        else: raise ValueError("Invalid model name.")
        
        model = model_builder(best_hps, vocab_size, max_len)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    if not os.path.exists('trained_models'): os.makedirs('trained_models')
    checkpoint_path = f"trained_models/{model_name}_best.keras"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss')

    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[early_stopping, model_checkpoint]
    )
    end_time = time.time()
    
    # Calculate training time
    training_time = round(end_time - start_time, 2)
    print(f"\n--- Training Complete for {model_name.upper()} in {training_time} seconds ---")
    return { "model": model, "history": history, "training_time": training_time, "model_path": checkpoint_path }