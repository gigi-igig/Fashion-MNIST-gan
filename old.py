import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

BUFFER_SIZE = 60000
BATCH_SIZE = 128


(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32")
train_images = (train_images - 127.5) / 127.5

train_labels = train_labels.astype(np.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def make_generator_model():
    noise_dim = 100
    num_classes = 10
    label_embedding_dim = 50

    noise_input = layers.Input(shape=(noise_dim,))
    label_input = layers.Input(shape=(1,), dtype='int32')

    # 1. 將 label 嵌入向量
    label_embedding = layers.Embedding(num_classes, label_embedding_dim)(label_input)
    label_embedding = layers.Flatten()(label_embedding)

    # 2. 合併 noise 和 label 向量
    combined_input = layers.Concatenate()([noise_input, label_embedding])

    # 3. 建立 Generator 網路
    x = layers.Dense(7*7*256, use_bias=False)(combined_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((7, 7, 256))(x)

    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                               use_bias=False, activation='tanh')(x)

    return tf.keras.Model([noise_input, label_input], x)

def make_discriminator_model():
    num_classes = 10
    label_embedding_dim = 50

    image_input = layers.Input(shape=(28, 28, 1))
    label_input = layers.Input(shape=(1,), dtype='int32')

    # 1. 將 label 轉為影像大小
    label_embedding = layers.Embedding(num_classes, 28 * 28)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape((28, 28, 1))(label_embedding)

    # 2. 將圖片和 label 疊在一起 (channels=2)
    combined_input = layers.Concatenate(axis=-1)([image_input, label_embedding])

    # 3. 建立 CNN 網路
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(combined_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model([image_input, label_input], x)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# 模型
generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

noise_dim = 100
num_examples_to_generate = 10

# 固定測試 label（0~9 循環）
seed = tf.random.normal([num_examples_to_generate, noise_dim])
seed_labels = tf.convert_to_tensor(np.array([i % 10 for i in range(num_examples_to_generate)]), dtype=tf.int32)

# ✅ Conditional GAN 訓練步驟
@tf.function
def train_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss 


def generate_and_save_images(model, epoch, test_input, test_labels):
    predictions = model([test_input, test_labels], training=False)
    fig = plt.figure(figsize=(15, 6))
    plt.subplots_adjust(hspace=0.5)  # 增加垂直間距

    for i in range(predictions.shape[0]):
        plt.subplot(2, 5, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')

        # 標題：顯示 label（自動支援 one-hot 或非 one-hot）
        label_value = tf.argmax(test_labels[i]).numpy() if len(test_labels[i].shape) > 0 else test_labels[i].numpy()
        plt.title(f"{label_value}", fontsize=10)
        plt.axis('off')

    plt.savefig(f'/kaggle/working/image_epoch_{epoch:03d}.png')
    plt.close()


train_log = []
def train(dataset, epochs, start_epoch=1):
    for epoch in range(epochs):
        start = time.time()
        gen_losses = []
        disc_losses = []

        for image_batch, label_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, label_batch)
            if (epoch + start_epoch) % 5 == 0:    
                gen_losses.append(gen_loss)
                disc_losses.append(disc_loss)

        epoch_time = time.time() - start
        print(f"[Epoch {epoch + start_epoch}] Time: {epoch_time:.2f}s")

        if (epoch + start_epoch) % 5 == 0:
            avg_gen_loss = tf.reduce_mean(gen_losses).numpy()
            avg_disc_loss = tf.reduce_mean(disc_losses).numpy()
            
            # 📌 紀錄 log
            train_log.append({
                'epoch': epoch + start_epoch,
                'gen_loss': avg_gen_loss,
                'disc_loss': avg_disc_loss,
                'time_sec': epoch_time
            })
            generate_and_save_images(generator, epoch + start_epoch, seed, seed_labels)
            generator.save(f"/kaggle/working/generator_epoch_{epoch+start_epoch:03d}.keras")
            discriminator.save(f"/kaggle/working/discriminator_epoch_{epoch + start_epoch:03d}.keras")

    # 📌 儲存 log 為 CSV
    df_log = pd.DataFrame(train_log)
    df_log.to_csv("/kaggle/working/training_log.csv", index=False)