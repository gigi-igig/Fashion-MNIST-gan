import tensorflow as tf
from tensorflow.keras import layers

class ConditionalGenerator(tf.keras.Model):
    def __init__(self, noise_dim=100, num_classes=10, label_embedding_dim=50):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_embedding = layers.Embedding(num_classes, label_embedding_dim)
        self.flatten = layers.Flatten()

        self.concat_dense = layers.Dense(7*7*256, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.lrelu1 = layers.LeakyReLU()
        self.reshape = layers.Reshape((7, 7, 256))

        self.deconv1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.lrelu2 = layers.LeakyReLU()

        self.deconv2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.lrelu3 = layers.LeakyReLU()

        self.deconv3 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                              use_bias=False, activation='tanh')

    def call(self, inputs, training=False):
        noise_input, label_input = inputs
        label_embedding = self.label_embedding(label_input)
        label_embedding = self.flatten(label_embedding)

        x = tf.concat([noise_input, label_embedding], axis=1)
        x = self.concat_dense(x)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.bn2(x, training=training)
        x = self.lrelu2(x)

        x = self.deconv2(x)
        x = self.bn3(x, training=training)
        x = self.lrelu3(x)

        return self.deconv3(x)
