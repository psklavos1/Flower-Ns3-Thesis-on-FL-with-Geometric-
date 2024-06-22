import tensorflow as tf

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# Check if TensorFlow is using the GPU
if gpus:
    print("TensorFlow is using the GPU")
else:
    print("TensorFlow is not using the GPU")

# Run a simple computation to see if it is placed on the GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result:\n", c)