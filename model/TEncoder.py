import keras
import tensorflow as tf
import numpy as np


def positional_encoding(lenght:int, depth:int):

    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(lenght)[:, np.newaxis],
                            np.arange(depth)[np.newaxis, :],
                            depth)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(keras.layers.Layer):
    
    def __init__(self, vocab_size:int, depth:int, lenght:int = 512):
        super().__init__()

        self.embedding = keras.layers.Embedding(vocab_size, depth)
        self.lenght = lenght
        self.depth = depth
    
    def call(self, x):

        x = self.embedding(x)
        return x + positional_encoding(self.lenght, self.depth)
    

class Attention(keras.layers.Layer):

    def __init__(self, dropout:float = 0.2):
        super().__init__()

        self.attention = keras.layers.Attention(
            dropout = dropout
        )

        self.normalize = keras.layers.LayerNormalization(epsilon = 10e-6)
        self.add = keras.layers.Add()

        self.attention_scores = None
    
    def call(self, x):

        out, attention_scores = self.attention(
            inputs = [x, x, x],
            return_attention_scores = True
        )
        self.attention_scores = attention_scores

        x = self.normalize(self.add([out, x]))

        return x
    

class MultiHeadAttention(keras.layers.Layer):

    def __init__(self, n_heads:int, 
                       depth:int, 
                       dropout:float = 0.2):
        super().__init__()

        self.attention = keras.layers.MultiHeadAttention(
            num_heads = n_heads,
            key_dim = depth,
            dropout = dropout
        )

        self.normalize = keras.layers.LayerNormalization(epsilon = 10e-6)
        self.add = keras.layers.Add()

        self.attention_scores = None
    
    def call(self, x):

        out, attention_scores = self.attention(
            key = x,
            value = x,
            query = x,
            return_attention_scores = True
        )
        self.attention_scores = attention_scores

        x = self.normalize(self.add([out, x]))

        return x


class FeedForward(keras.layers.Layer):

    def __init__(self, depth:int,
                       grandezza_feed:int, 
                       dropout:float = 0.2):
        super().__init__()

        self.feed_forward = keras.models.Sequential([
            keras.layers.Dense(grandezza_feed, activation='relu'),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(depth)
        ])

        self.add = keras.layers.Add()
        self.normalize = keras.layers.LayerNormalization()


    def call(self, x):

        p = self.feed_forward(x)

        x = self.add([x, p])
        x = self.normalize(x)

        return x
    
class EncoderLayer(keras.layers.Layer):

    def __init__(self, *, grandezza_feed:int, 
                          depth:int | None = None,
                          n_teste:int | None = None, 
                          type_enc:str = 'dot',
                          dropout:float = 0.2):
        super().__init__()

        match type_enc:

            case 'mha':
                self.attention = MultiHeadAttention(
                    n_heads = n_teste,
                    depth = depth,
                    dropout = dropout
                )

            case 'dot':
                self.attention = Attention(dropout)

        self.feed_forward = FeedForward(depth, grandezza_feed, dropout)
    
    def call(self, x):

        x = self.attention(x)
        x = self.feed_forward(x)

        return x
    

class TransformerEncoder(keras.models.Model):

    def __init__(self, *, depth:int, 
                          numero_layers:int,
                          grandezza_vocabolario:int,
                          n_teste:int | None = None, 
                          grandezza_feed:int | None = None,
                          type_enc:str = 'dot',
                          dropout:float = 0.2):
        super().__init__()
        
        self.grandezza_vocabolario = grandezza_vocabolario
        self.depth = depth
        self.numero_layers = numero_layers

        self.encoding_layers = [

            EncoderLayer(
                    depth = depth, 
                    grandezza_feed = grandezza_feed,
                    n_teste = n_teste,
                    dropout = dropout,
                    type_enc = type_enc)

            for _ in range(numero_layers)
        ]

        self.dropout = keras.layers.Dropout(dropout)

    def build(self, input_shape):

        self.embedding = PositionalEncoding(self.grandezza_vocabolario, 
                                            self.depth,
                                            input_shape[1])

    def call(self, x):
        
        x = self.embedding(x)
        x = self.dropout(x)

        for layer_encoding in self.encoding_layers:
            x = layer_encoding(x)

        return x