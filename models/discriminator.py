import tensorflow as tf
from tensorflow.keras import layers

class ConditionalDiscriminator(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.label_embedding = layers.Embedding(num_classes, 28 * 28)
        self.flatten = layers.Flatten()
        self.reshape = layers.Reshape((28, 28, 1))

        self.conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.lrelu1 = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.3)

        self.conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.lrelu2 = layers.LeakyReLU()
        self.dropout2 = layers.Dropout(0.3)

        self.flatten_final = layers.Flatten()
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=False):
        image_input, label_input = inputs
        label_embedding = self.label_embedding(label_input)
        label_embedding = self.flatten(label_embedding)
        label_embedding = self.reshape(label_embedding)

        x = tf.concat([image_input, label_embedding], axis=-1)

        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.dropout2(x, training=training)

        x = self.flatten_final(x)
        return self.output_layer(x)
