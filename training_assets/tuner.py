import keras_tuner as kt
from models.lstm_model import build_lstm_model
from models.gru_model import build_gru_model
from models.transformer_model import build_transformer_model
from metrics import perplexity

def find_best_hyperparameters(strategy, model_name: str, train_ds, val_ds, vocab_size, max_len):
    """
    Finds the best hyperparameters for a given model using Keras Tuner on multiple GPUs.

    Args:
        strategy: The TensorFlow distribution strategy (e.g., MirroredStrategy).
        model_name (str): The name of the model ('lstm', 'gru', 'transformer').
        train_ds, val_ds: The TF Dataset objects for training and validation.
        vocab_size (int): Vocabulary size.
        max_len (int): Max sequence length.

    Returns:
        A Keras Tuner HyperParameters object with the best found parameters.
    """
    print(f"\n--- Starting Multi-GPU Hyperparameter Tuning for {model_name.upper()} ---")

    if model_name == 'lstm':
        build_fn = build_lstm_model
    elif model_name == 'gru':
        build_fn = build_gru_model
    elif model_name == 'transformer':
        build_fn = build_transformer_model
    else:
        raise ValueError("Invalid model name.")

    def model_builder_scoped(hp):
        with strategy.scope():
            model = build_fn(hp, vocab_size, max_len)
        return model

    tuner = kt.RandomSearch(
        hypermodel=model_builder_scoped,
        objective=kt.Objective("val_perplexity", direction="min"),
        max_trials=3,
        executions_per_trial=1,
        directory='tuning_results',
        project_name=f'{model_name}_tuner_multi_gpu'
    )

    print(f"Searching for best hyperparameters across {strategy.num_replicas_in_sync} GPUs...")
    tuner.search(train_ds, epochs=5, validation_data=val_ds)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"\n--- Tuning Complete for {model_name.upper()} ---")
    print(f"Optimal Embedding Dim: {best_hps.get('embedding_dim') or best_hps.get('embed_dim')}")
    print(f"Optimal Learning Rate: {best_hps.get('learning_rate')}")

    return best_hps