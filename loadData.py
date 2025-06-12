import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from config import (
    BUFFER_SIZE, BATCH_SIZE, NOISE_DIM, NUM_EXAMPLES_TO_GENERATE,
    SAVE_INTERVAL, MODEL_SAVE_PATH, DISC_SAVE_PATH,
    IMAGE_SAVE_PATH, LOG_SAVE_PATH
)

def load_data(OVERSAMPLE_FACTOR = 1, TARGET_LABELS=[]):

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32")
    train_images = (train_images - 127.5) / 127.5

    train_labels = train_labels.astype(np.int32)

    test_images = (test_images.reshape((-1, 28, 28, 1)).astype("float32") - 127.5) / 127.5
    test_labels = test_labels.astype(np.int32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    if OVERSAMPLE_FACTOR == 1:

        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, test_dataset

    # 建立原始資料集
    original_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

    # 過濾與擴增特定 label
    def oversample_class(images, labels, target_label, factor):
        # 過濾該類別
        mask = tf.where(tf.equal(labels, target_label))
        target_images = tf.gather(images, mask[:, 0])
        target_labels = tf.gather(labels, mask[:, 0])

        # 複製多次
        repeated_images = tf.repeat(target_images, repeats=[factor - 1], axis=0)
        repeated_labels = tf.repeat(target_labels, repeats=[factor - 1], axis=0)
        
        return tf.data.Dataset.from_tensor_slices((repeated_images, repeated_labels))
    
    # 建立所有 oversampled 子資料集
    oversampled_ds_list = [oversample_class(train_images, train_labels, label, OVERSAMPLE_FACTOR)
                        for label in TARGET_LABELS]
    # 合併所有資料集
    full_ds = original_ds.concatenate(tf.data.Dataset.from_tensor_slices((train_images, train_labels)))
    for ds in oversampled_ds_list:
        full_ds = full_ds.concatenate(ds)


    # 資料增強方法
    def augment_image(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image

    def augment_if_target(image, label):
        should_augment = tf.reduce_any(tf.equal(label, TARGET_LABELS))
        image = tf.cond(should_augment, lambda: augment_image(image), lambda: image)
        return image, label

    train_dataset = (
        full_ds
        .shuffle(BUFFER_SIZE)
        .map(augment_if_target, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, test_dataset