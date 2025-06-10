import tensorflow as tf
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_labels = tf.random.uniform(shape=tf.shape(real_output), minval=0.8, maxval=1.0)
    fake_labels = tf.random.uniform(shape=tf.shape(fake_output), minval=0.0, maxval=0.2)
    
    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)