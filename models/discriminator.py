import tensorflow as tf
from tensorflow.keras import layers

@register_keras_serializable()
class MultiBranchConditionalDiscriminator(tf.keras.Model):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.shared_conv1 = layers.Conv2D(64, (5,5), strides=(2,2), padding='same')
        self.shared_lrelu1 = layers.LeakyReLU(0.2)
        self.shared_conv2 = layers.Conv2D(128, (5,5), strides=(2,2), padding='same')
        self.shared_bn2 = layers.BatchNormalization()
        self.shared_lrelu2 = layers.LeakyReLU(0.2)
        self.shared_flatten = layers.Flatten()

        self.class_branches = []
        for _ in range(num_classes):
            branch = tf.keras.Sequential([
                layers.Dense(64),
                layers.LeakyReLU(0.2),
                layers.Dense(1)
            ])
            self.class_branches.append(branch)

    def call(self, inputs, training=True):
        images, labels = inputs

        x = self.shared_conv1(images)
        x = self.shared_lrelu1(x)
        x = self.shared_conv2(x)
        x = self.shared_bn2(x, training=training)
        x = self.shared_lrelu2(x)
        x = self.shared_flatten(x)

        outputs = []
        for class_id in range(self.num_classes):
            mask = tf.equal(labels, class_id)
            indices = tf.where(mask)[:, 0]
            selected_x = tf.gather(x, indices)

            def empty_fn():
                return tf.zeros((0, 1), dtype=tf.float32)
            def branch_fn():
                return self.class_branches[class_id](selected_x, training=training)

            out = tf.cond(tf.shape(selected_x)[0] > 0, branch_fn, empty_fn)
            outputs.append(out)

        final_output = tf.concat(outputs, axis=0)
        return final_output

    def build(self, input_shape=None):
        if input_shape is None:
            input_shape = [(None, 28, 28, 1), (None,)]
    
        dummy_img = tf.zeros((1, 28, 28, 1))
        dummy_label = tf.zeros((1,), dtype=tf.int32)
        
        # 使用 call() 而不是 self(...)，避免觸發建構流程
        super().build(input_shape)
        _ = self.call((dummy_img, dummy_label), training=False)
