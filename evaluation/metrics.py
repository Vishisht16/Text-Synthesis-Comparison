import tensorflow as tf
from tensorflow.keras import backend as K

@tf.keras.saving.register_keras_serializable()
def perplexity(y_true, y_pred):
    """
    Custom Keras metric to calculate perplexity.
    """
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.exp(K.mean(cross_entropy))