import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from metrics import perplexity
from tensorflow.keras import mixed_precision

# Transformer Block and Token Embedding Layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Transformer Model Builder
def build_transformer_model(hp, vocab_size, max_len):
    """
    Builds a Transformer-based model for Keras Tuner.
    
    Args:
        hp: Keras Tuner HyperParameters object.
        vocab_size (int): The size of the vocabulary.
        max_len (int): The max sequence length for input.
    
    Returns:
        A compiled Keras model.
    """
    embed_dim = hp.Int('embedding_dim', min_value=64, max_value=128, step=32)
    
    # Ensure embed_dim is even and a multiple of 4 for MultiHeadAttention
    if embed_dim % 2 != 0: embed_dim += 1
    if embed_dim % 4 != 0: embed_dim = (embed_dim // 4 + 1) * 4
    
    num_heads = hp.Choice('num_heads', values=[2, 4]) 
    ff_dim = hp.Int('ff_dim', min_value=128, max_value=256, step=64)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    
    inputs = layers.Input(shape=(max_len - 1,))
    
    # 1. Embedding Layer
    embedding_layer = TokenAndPositionEmbedding(max_len - 1, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    
    # 2. Transformer Block
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate=dropout_rate)
    x = transformer_block(x)

    # 3. Output Processing
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy', perplexity],
        jit_compile=True # Enable XLA for performance
    )
    
    return model