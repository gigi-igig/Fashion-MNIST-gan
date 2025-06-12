import tensorflow as tf
from tensorflow.keras import layers

class SpectralNormalization(layers.Wrapper):
    def __init__(self, layer, power_iterations=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.random_normal_initializer(),
            trainable=False,
            name='sn_u'
        )
        super(SpectralNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(w_reshaped)))
            u = tf.math.l2_normalize(tf.matmul(v, w_reshaped))
        sigma = tf.matmul(tf.matmul(v, w_reshaped), tf.transpose(u))
        w_norm = self.w / sigma
        self.w.assign(w_norm) 

        self.u.assign(u)
        return self.layer(inputs)
