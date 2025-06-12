
import tensorflow as tf
from tensorflow.keras import layers

@register_keras_serializable()
class MultiBranchConditionalGenerator(tf.keras.Model):
    def __init__(self, noise_dim=100, num_classes=10, label_embedding_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.label_embedding_dim = label_embedding_dim

        self.label_embedding = layers.Embedding(num_classes, label_embedding_dim)
        self.flatten = layers.Flatten()
        initializer = tf.keras.initializers.GlorotUniform()

        self.concat_dense = layers.Dense(7*7*256, use_bias=False, kernel_initializer=initializer)
        self.bn1 = layers.BatchNormalization()
        self.lrelu1 = layers.LeakyReLU()
        self.reshape = layers.Reshape((7, 7, 256))

        self.class_branches = []
        for _ in range(num_classes):
            branch = tf.keras.Sequential([
                layers.Conv2DTranspose(256, (5,5), strides=(1,1), padding='same', use_bias=False, kernel_initializer=initializer),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Dropout(0.3),
                layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False, kernel_initializer=initializer),
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh', kernel_initializer=initializer),
            ])
            self.class_branches.append(branch)

    def call(self, inputs, training=False):
        noise_input, label_input = inputs
        label_embedding = self.label_embedding(label_input)
        label_flat = self.flatten(label_embedding)

        x = tf.concat([noise_input, label_flat], axis=1)
        x = self.concat_dense(x)
        x = self.bn1(x, training=training)
        x = self.lrelu1(x)
        x = self.reshape(x)

        label_broadcast = tf.expand_dims(tf.expand_dims(label_embedding, 1), 1)
        label_broadcast = tf.tile(label_broadcast, [1, 7, 7, 1])
        x = tf.concat([x, label_broadcast], axis=-1)

        partitions = label_input
        x_parts = tf.dynamic_partition(x, partitions, self.num_classes)
        idx_parts = tf.dynamic_partition(tf.range(tf.shape(label_input)[0]), partitions, self.num_classes)

        outputs_parts = []
        for class_id in range(self.num_classes):
            part = x_parts[class_id]
            def branch_call():
                return self.class_branches[class_id](part, training=training)
            def empty_call():
                return tf.zeros((0, 28, 28, 1), dtype=tf.float32)
            out = tf.cond(tf.shape(part)[0] > 0, branch_call, empty_call)
            outputs_parts.append(out)

        output = tf.dynamic_stitch(idx_parts, outputs_parts)
        return output

    def build(self, input_shape):
        noise_shape, label_shape = input_shape
        
        # 初始化嵌入層與展平層
        self.label_embedding.build(label_shape)
        label_embed_output_shape = self.label_embedding.compute_output_shape(label_shape)
        self.flatten.build(label_embed_output_shape)
    
        # concat後的維度 = noise_dim + label_embedding_dim
        concat_input_dim = self.noise_dim + self.label_embedding_dim
        self.concat_dense.build((None, concat_input_dim))
        self.bn1.build((None, 7*7*256))
        self.lrelu1.build((None, 7*7*256))
        self.reshape.build((None, 7*7*256))
    
        # 為每個 class branch 都明確建構（假設輸入 shape 是 [None, 7, 7, 256 + label_embedding_dim]）
        for branch in self.class_branches:
            branch.build((None, 7, 7, 256 + self.label_embedding_dim))
    
        super().build(input_shape)
