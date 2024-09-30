import tensorflow as tf 

import keras


class Autocorrelation(keras.layers.Layer):

    def __init__(self, depth_model:int, autocorrelation_factor:int = 2):
        super().__init__()

        self.autocorrelation_factor = autocorrelation_factor

        self.key_linear = keras.layers.Dense(depth_model)
        self.query_linear = keras.layers.Dense(depth_model)
        self.value_linear = keras.layers.Dense(depth_model)

        self.last_layer = keras.layers.Dense(depth_model)

    def build(self, input_shape):
        '''
        Necessary funcion in order to let tensorflow automatically initialize
        the autocorrelation preprocessing layers
        '''
        return

    def irfft_with_axis(self, array:tf.Tensor, axis:int):
    
        perm = list(range(len(array.shape)))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        transposed_states = tf.transpose(array, perm)

        irfft_result = tf.signal.irfft(transposed_states)

        irfft_result = tf.transpose(irfft_result, perm)

        return irfft_result
    

    def rfft_with_axis(self, array:tf.Tensor, axis:int):
        
        perm = list(range(len(array.shape)))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        transposed_states = tf.transpose(array, perm)

        rfft_result = tf.signal.rfft(transposed_states)

        rfft_result = tf.transpose(rfft_result, perm)

        return rfft_result
    

    def top_k_with_axis(self, array:tf.Tensor, k:int, axis:int):
        
        perm = list(range(len(array.shape)))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        transposed_states = tf.transpose(array, perm)

        top_k_result, top_k_indexes = tf.math.top_k(transposed_states, k)

        top_k_result = tf.transpose(top_k_result, perm)
        top_k_indexes = tf.transpose(top_k_indexes, perm)

        return top_k_result, top_k_indexes
    

    def autocorrelation(self, query_states:tf.Tensor, key_states:tf.Tensor):

        query_states_fft = self.rfft_with_axis(query_states, 1)
        key_states_fft = self.rfft_with_axis(key_states, 1)

        attn_weights = query_states_fft * tf.math.conj(key_states_fft)

        attn_weights = self.irfft_with_axis(attn_weights, 1)

        return attn_weights
    
    @tf.function
    def time_delay_aggregation(self, attn_weights:tf.Tensor, 
                                     value_states:tf.Tensor):

        time_len = tf.shape(value_states)[1]
        
        top_k = self.autocorrelation_factor * tf.math.log(tf.cast(time_len, tf.float32))
        top_k = tf.cast(top_k, tf.int32)

        autocorrelation_mean = tf.reduce_mean(attn_weights, -1)
        topk_autocorrelations, topk_delays = self.top_k_with_axis(autocorrelation_mean, top_k, axis=1)

        autocorrelation_result = tf.TensorArray(tf.float32, size=tf.shape(value_states)[0])
        for batch in tf.range(tf.shape(value_states)[0]):

            temp_result = tf.zeros_like(value_states[batch])
            for i in tf.range(top_k):

                values_rolled = tf.roll(value_states[batch], topk_delays[batch][i], 0)
                soft_attn = tf.nn.softmax(attn_weights[batch], axis=-1)

                temp_result = temp_result + values_rolled * soft_attn
            
            autocorrelation_result = autocorrelation_result.write(batch, temp_result)

        autocorrelation_result = autocorrelation_result.stack()

        return autocorrelation_result
    

    def call(self, x:tf.Tensor | list, training:bool = False):

        query, key, value = x

        if key.shape[1] != query.shape[1]:

            idx = key.shape[1] - query.shape[1]

            key = key[:, idx:, :]
            value = value[:, idx:, :]


        query = self.query_linear(query, training = training)
        key = self.key_linear(key, training = training)
        value = self.value_linear(value, training = training)

        attn_weights = self.autocorrelation(query, key)
        attn_output = self.time_delay_aggregation(attn_weights, value)

        out = self.last_layer(attn_output)

        return out


class SeriesDecomposition(keras.layers.Layer):

    def __init__(self, kernel_size:int = 25):
        super().__init__()

        self.avg_pool = keras.layers.AveragePooling1D(kernel_size, strides=1, padding='same')

    def call(self, x, training: bool = False):

        trend = self.avg_pool(x, training = training)
        seasonal = x - trend

        return trend, seasonal


class FeedForward(keras.layers.Layer):

    def __init__(self, depth_model:int, size_net:int):
        super().__init__()

        self.hidden_layer = keras.layers.Dense(size_net, activation = 'relu')
        self.out_layer = keras.layers.Dense(depth_model)

    def call(self, x):

        x = self.hidden_layer(x)
        x = self.out_layer(x)

        return x


class EncoderBlock(keras.layers.Layer):

    def __init__(self, depth_model:int, 
                       size_net:int = 256,
                       autocorrelation_factor:int = 2,
                       kernel_size:int = 25):
        super().__init__()

        self.autocorrelation = Autocorrelation(depth_model, autocorrelation_factor)

        self.add_1 = keras.layers.Add()
        self.add_2 = keras.layers.Add()

        self.decomposition_1 = SeriesDecomposition(kernel_size)
        self.decomposition_2 = SeriesDecomposition(kernel_size)

        self.preprocess = FeedForward(depth_model, size_net)
        self.feed_forward = FeedForward(depth_model, size_net)

    def call(self, x):

        x = self.preprocess(x)

        autocorrelation_out = self.autocorrelation([x, x, x])
        x = self.add_1([autocorrelation_out, x])

        _, seasonal = self.decomposition_1(x)

        feed_out = self.feed_forward(seasonal)
        x = self.add_2([feed_out, seasonal])

        _, seasonal_out = self.decomposition_2(x)

        return seasonal_out
    

class AutoformerEncoder(keras.layers.Layer):

    def __init__(self, depth_model:int, 
                       num_encoders:int,
                       size_net:int = 256,
                       autocorrelation_factor:int = 2,
                       kernel_size:int = 25):
        super().__init__()

        self.model = keras.models.Sequential([
            EncoderBlock(
                depth_model = depth_model,
                size_net = size_net,
                autocorrelation_factor = autocorrelation_factor,
                kernel_size = kernel_size
            ) 
            
            for _ in range(num_encoders)
        ])
        

    def call(self, x):

        if tf.rank(x) == 2:

            return self.model(tf.expand_dims(x, axis=0))

        elif tf.rank(x) == 3:
            return self.model(x)
        
        else:
            raise ValueError('Prices rank must be in the range [2, 3]')