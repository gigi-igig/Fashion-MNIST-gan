import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from models.generator import MultiBranchConditionalGenerator
from models.discriminator import MultiBranchConditionalDiscriminator
from utils.image_utils import generate_and_save_images
from utils.losses import generator_loss, discriminator_loss
from loadData import load_data
from config import (
    BUFFER_SIZE, BATCH_SIZE, NOISE_DIM, NUM_EXAMPLES_TO_GENERATE,
    SAVE_INTERVAL, MODEL_SAVE_WEIGHT_PATH, DISC_SAVE_WEIGHT_PATH,
    IMAGE_SAVE_PATH, LOG_SAVE_PATH
)

train_dataset, test_dataset = load_data()

# 建立模型
# 初始化模型
generator = MultiBranchConditionalGenerator(num_classes=10, noise_dim=NOISE_DIM)
generator.build([(None, NOISE_DIM), (None,)])
discriminator = MultiBranchConditionalDiscriminator(num_classes=10)
discriminator.build([(None, 28, 28, 1), (None,)])

# 優化器
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


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
        disc_loss = discriminator_loss(real_output, fake_output, smooth=True)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs, start_epoch=1):

    train_log = []

    for epoch in range(epochs):
        start_time = time.time()
        gen_losses = []
        disc_losses = []

        for image_batch, label_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, label_batch)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        epoch_time = time.time() - start_time
        avg_gen_loss = tf.reduce_mean(gen_losses).numpy()
        avg_disc_loss = tf.reduce_mean(disc_losses).numpy()

        print(f"Epoch {epoch + start_epoch}, gen_loss: {avg_gen_loss:.4f}, disc_loss: {avg_disc_loss:.4f}, time: {epoch_time:.2f}s")

        train_log.append({
            'epoch': epoch + start_epoch,
            'gen_loss': avg_gen_loss,
            'disc_loss': avg_disc_loss,
            'time_sec': epoch_time
        })

        # 每隔 SAVE_INTERVAL 儲存模型權重與生成圖片
        if (epoch + start_epoch) % SAVE_INTERVAL == 0:
            generate_and_save_images(generator, epoch + start_epoch, seed, seed_labels, IMAGE_SAVE_PATH)
           
            generator.save_weights( MODEL_SAVE_WEIGHT_PATH.format(epoch + start_epoch))
            discriminator.save_weights(DISC_SAVE_WEIGHT_PATH.format(epoch + start_epoch))

    pd.DataFrame(train_log).to_csv(LOG_SAVE_PATH, index=False)

if __name__ == "__main__":
    train(train_dataset, epochs=10)
