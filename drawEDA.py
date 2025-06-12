from sklearn.manifold import TSNE
from collections import Counter
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config import (
    BUFFER_SIZE, BATCH_SIZE, NOISE_DIM, NUM_EXAMPLES_TO_GENERATE,
    SAVE_INTERVAL,IMAGE_SAVE_PATH, LOG_SAVE_PATH
)

(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = (train_images.reshape((-1, 28, 28, 1)).astype("float32")- 127.5) / 127.5
train_labels = train_labels.astype(np.int32)

def draw_image(images, image_id):
    plt.imshow(images[image_id], cmap='gray')
    plt.title(f"Label: {train_labels[0]}")
    plt.axis('off')
    plt.show()
    

def draw_counts(labels):
    # 計算每個類別的樣本數
    label_counts = Counter(labels)
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

def draw_tSNE(images, labels):
    # 選前 2000 筆資料進行 t-SNE（否則太慢）
    subset_images = images[:2000]
    subset_labels = labels[:2000].astype(int)

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


# 建立每個 label 的平均圖像
def draw_correlation(images, labels, max_per_class=2000):
    mean_images = []
    for label in range(10):
        # 擷取該類別前 max_per_class 張圖片
        class_images = images[labels == label][:max_per_class]
        mean_img = np.mean(class_images, axis=0).flatten()
        mean_images.append(mean_img)

    # 計算 label 間的相關矩陣（Pearson correlation）
    mean_images = np.array(mean_images)
    correlation_matrix = np.corrcoef(mean_images)

    # 顯示相關矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=range(10), yticklabels=range(10))
    plt.title("Correlation between average images of labels")
    plt.show()

#draw_image(train_images, 0)
#draw_counts(train_labels)
#draw_tSNE(train_images, train_labels)
draw_correlation(train_images, train_labels)