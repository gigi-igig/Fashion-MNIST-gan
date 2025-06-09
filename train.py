import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from models.generator import ConditionalGenerator
from models.discriminator import ConditionalDiscriminator
from utils.image_utils import generate_and_save_images
from utils.losses import generator_loss, discriminator_loss
from config import (
    BUFFER_SIZE, BATCH_SIZE, NOISE_DIM, NUM_EXAMPLES_TO_GENERATE,
    SAVE_INTERVAL, MODEL_SAVE_PATH, DISC_SAVE_PATH,
    IMAGE_SAVE_PATH, LOG_SAVE_PATH
)


(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32")
train_images = (train_images - 127.5) / 127.5

train_labels = train_labels.astype(np.int32)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 建立模型
generator = ConditionalGenerator()
discriminator = ConditionalDiscriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

# 測試用 latent noise 與 label
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])
seed_labels = tf.convert_to_tensor(np.array([i % 10 for i in range(NUM_EXAMPLES_TO_GENERATE)]), dtype=tf.int32)

train_log = []

# Conditional GAN 訓練步驟
@tf.function
def train_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

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


def train(dataset, epochs, start_epoch=1):
    for epoch in range(epochs):
        start = time.time()
        gen_losses = []
        disc_losses = []

        for image_batch, label_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, label_batch)
            if (epoch + start_epoch) % SAVE_INTERVAL == 0:
                gen_losses.append(gen_loss)
                disc_losses.append(disc_loss)

        epoch_time = time.time() - start
        print(f"[Epoch {epoch + start_epoch}] Time: {epoch_time:.2f}s")

        if (epoch + start_epoch) % SAVE_INTERVAL == 0:
            avg_gen_loss = tf.reduce_mean(gen_losses).numpy()
            avg_disc_loss = tf.reduce_mean(disc_losses).numpy()

            train_log.append({
                'epoch': epoch + start_epoch,
                'gen_loss': avg_gen_loss,
                'disc_loss': avg_disc_loss,
                'time_sec': epoch_time
            })
            generate_and_save_images(generator, epoch + start_epoch, seed, seed_labels, IMAGE_SAVE_PATH)
            generator.save(MODEL_SAVE_PATH.format(epoch + start_epoch))
            discriminator.save(DISC_SAVE_PATH.format(epoch + start_epoch))

    df_log = pd.DataFrame(train_log)
    df_log.to_csv(LOG_SAVE_PATH, index=False)

if __name__ == "__main__":
    train(train_dataset, epochs=50)
