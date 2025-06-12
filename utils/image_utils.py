import matplotlib.pyplot as plt
import tensorflow as tf


def generate_and_save_images(model, epoch, test_input, test_labels, path):
    predictions = model([test_input, test_labels], training=False)
    fig = plt.figure(figsize=(15, 6))
    plt.subplots_adjust(hspace=0.5)

    for i in range(predictions.shape[0]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')

        label_value = (
            tf.argmax(test_labels[i]).numpy()
            if len(test_labels[i].shape) > 0
            else test_labels[i].numpy()
        )
        plt.title(f"{label_value}", fontsize=10)
        plt.axis("off")

    plt.savefig(path.format(epoch))
    plt.close()

def generate_and_save_images(model, epoch, test_input, test_labels, path):
    predictions = model((test_input, test_labels), training=False)
    predictions = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)

    fig = plt.figure(figsize=(10,1))
    for i in range(predictions.shape[0]):
        plt.subplot(1, NUM_EXAMPLES_TO_GENERATE, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(path.format(epoch))
    plt.close()