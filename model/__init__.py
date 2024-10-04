import keras
import tensorflow as tf 

from model.AEncoder import AutoformerEncoder
from model.TEncoder import TransformerEncoder


class NewsProcessing(keras.layers.Layer):

    def __init__(self, grandezza_vocabolario_titles: int,
                       grandezza_vocabolario_paragraphs: int,

                       depth_titles:int = 256,
                       depth_paragraphs:int = 256,
                       numero_layers_titles:int = 5,
                       numero_layers_paragraphs:int = 5,
           
                       n_teste_titles:int | None = 8,
                       n_teste_paragraphs:int | None = 8,
                       grandezza_feed_titles: int | None = 2048,
                       grandezza_feed_paragraphs: int | None = 2048,
           
                       type_enc_titles: str = 'mha',
                       type_enc_paragraphs: str = 'mha',
                       dropout_titles: float = 0.2,
                       dropout_paragraphs: float = 0.2):
        super().__init__()

        self.TEncoder_titles = TransformerEncoder(
            depth = depth_titles,
            numero_layers = numero_layers_titles,
            grandezza_vocabolario = grandezza_vocabolario_titles,
            n_teste = n_teste_titles,
            grandezza_feed = grandezza_feed_titles,
            type_enc = type_enc_titles,
            dropout = dropout_titles
        )

        self.TEncoder_paragraphs = TransformerEncoder(
            depth = depth_paragraphs,
            numero_layers = numero_layers_paragraphs,
            grandezza_vocabolario = grandezza_vocabolario_paragraphs,
            n_teste = n_teste_paragraphs,
            grandezza_feed = grandezza_feed_paragraphs,
            type_enc = type_enc_paragraphs,
            dropout = dropout_paragraphs
        )

        self.compat_attention = keras.layers.MultiHeadAttention(8, depth_paragraphs)


    def call(self, x):

        if x[0].ndim == 3:

            titles = tf.map_fn(
                self.TEncoder_titles,
                x[0],
                fn_output_signature = tf.float32
            )

            paragraphs = tf.map_fn(
                self.TEncoder_paragraphs,
                x[1],
                fn_output_signature = tf.float32
            )

            titles = tf.reduce_mean(titles, axis=1)
            paragraphs = tf.reduce_mean(paragraphs, axis=1)


        elif x[0].ndim == 2:

            titles = self.TEncoder_titles(x[0])
            paragraphs = self.TEncoder_paragraphs(x[1])

            titles = tf.expand_dims(tf.reduce_mean(titles, axis=0), axis=0)
            paragraphs = tf.expand_dims(tf.reduce_mean(paragraphs, axis=0), axis=0)


        else:

            raise ValueError('Rank of titles must be in the range of [2, 3]')


        x = self.compat_attention(
            query = titles,
            key = paragraphs,
            value = paragraphs
        )

        return x


class FinanceModel(keras.models.Model):

    def __init__(self, grandezza_vocabolario_titles: int,
                       grandezza_vocabolario_paragraphs: int,

                       depth_titles:int = 256,
                       depth_paragraphs:int = 256,
                       depth_prices:int = 256,
                       numero_layers_titles:int = 5,
                       numero_layers_paragraphs:int = 5,
                       numero_layers_prices:int = 4,
           
                       n_teste_titles:int | None = 8,
                       n_teste_paragraphs:int | None = 8,
                       grandezza_feed_titles: int | None = 2048,
                       grandezza_feed_paragraphs: int | None = 2048,
                       grandezza_feed_prices:int = 256,
                       grandezza_last_feed:int = 2048,
           
                       type_enc_titles: str = 'mha',
                       type_enc_paragraphs: str = 'mha',
                       autocorrelation_factor:int = 2,
                       kernel_size_prices:int = 25,
                       dropout_titles: float = 0.2,
                       dropout_paragraphs: float = 0.2):
        super().__init__()

        self.news_processing = NewsProcessing(
            grandezza_vocabolario_titles,
            grandezza_vocabolario_paragraphs,

            depth_titles,
            depth_paragraphs,
            numero_layers_titles,
            numero_layers_paragraphs,

            n_teste_titles,
            n_teste_paragraphs,
            grandezza_feed_titles,
            grandezza_feed_paragraphs,

            type_enc_titles,
            type_enc_paragraphs,
            dropout_titles,
            dropout_paragraphs
        )

        self.price_processing = AutoformerEncoder(
            depth_prices,
            numero_layers_prices,
            grandezza_feed_prices,
            autocorrelation_factor,
            kernel_size_prices
        )

        self.feed_out = keras.models.Sequential([
            keras.layers.Dense(grandezza_last_feed, activation='relu'),
            keras.layers.Dense(grandezza_last_feed // 2),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, x):

        titles, paragraphs, prices = x

        news_output = self.news_processing([titles, paragraphs])

        prices_output = self.price_processing(prices)

        out = tf.concat([news_output, prices_output], axis=1)
        out = self.feed_out(out)
        out = tf.reduce_mean(out, axis=1)

        return out