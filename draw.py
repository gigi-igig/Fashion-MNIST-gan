from sklearn.manifold import TSNE
from collections import Counter
import tensorflow as tf
import matplotlib as plt
import numpy as np
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

# 計算每個類別的樣本數
label_counts = Counter(train_labels)
print("每個類別的樣本數量：")
for label in range(10):
    print(f"Label {label}: {label_counts[label]}")

# 畫出直方圖
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Label Distribution in Fashion-MNIST")
plt.xticks(range(10))
plt.show()
# 選前 2000 筆資料進行 t-SNE（否則太慢）
subset_images = train_images[:2000]
subset_labels = train_labels[:2000].astype(int)

# 攤平成向量 (28*28)
flatten_images = subset_images.reshape((subset_images.shape[0], -1))

# t-SNE 降到 2 維
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(flatten_images)

# 畫圖（不同類別不同顏色）
plt.figure(figsize=(10, 7))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=subset_labels, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE of Fashion-MNIST (first 2000 images)")
plt.show()
