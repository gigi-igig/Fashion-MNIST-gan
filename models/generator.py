import tensorflow as tf
from tensorflow.keras import layers

class ConditionalGenerator(tf.keras.Model):
    def __init__(self, noise_dim=100, num_classes=10, label_embedding_dim=50):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_embedding = layers.Embedding(num_classes, label_embedding_dim)
        self.flatten = layers.Flatten()

        # 初始化器
        initializer = tf.keras.initializers.GlorotUniform()

        # Dense: 噪音 + label -> 7x7x256
        self.concat_dense = layers.Dense(7*7*256, use_bias=False, kernel_initializer=initializer)
        self.bn1 = layers.BatchNormalization()
        self.lrelu1 = layers.LeakyReLU()
        self.reshape = layers.Reshape((7, 7, 256))

        # Dropout 增加穩定性
        self.dropout = layers.Dropout(0.3)

        # Conv2DTranspose layers
        self.deconv1 = layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', 
                                              use_bias=False, kernel_initializer=initializer)
        self.bn2 = layers.BatchNormalization()
        self.lrelu2 = layers.LeakyReLU()

        self.deconv2 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', 
                                              use_bias=False, kernel_initializer=initializer)
        self.bn3 = layers.BatchNormalization()
        self.lrelu3 = layers.LeakyReLU()

        self.deconv3 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                              use_bias=False, activation='tanh',
                                              kernel_initializer=initializer)

    def call(self, inputs, training=False):
        noise_input, label_input = inputs

        # label -> embedding -> 展平
        label_embedding = self.label_embedding(label_input)
        label_flat = self.flatten(label_embedding)

        # 初始拼接：noise + label -> dense
        x = tf.concat([noise_input, label_flat], axis=1)
        x = self.concat_dense(x)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)
        x = self.reshape(x)

        # 將 label 展開為圖像大小，拼接至 feature map
        label_broadcast = tf.expand_dims(tf.expand_dims(label_embedding, 1), 1)  # (batch, 1, 1, emb_dim)
        label_broadcast = tf.tile(label_broadcast, [1, 7, 7, 1])  # (batch, 7, 7, emb_dim)
        x = tf.concat([x, label_broadcast], axis=-1)  # (batch, 7, 7, 256 + emb_dim)

        x = self.deconv1(x)
        x = self.bn2(x, training=training)
        x = self.lrelu2(x)
        x = self.dropout(x, training=training)

        x = self.deconv2(x)
        x = self.bn3(x, training=training)
        x = self.lrelu3(x)

        return self.deconv3(x)
