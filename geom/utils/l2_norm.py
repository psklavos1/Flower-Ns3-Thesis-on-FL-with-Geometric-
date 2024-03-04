# third party
import tensorflow as tf
import numpy as np
import copy

# To create graph functions for efficiency
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


class L2_norm(tf.keras.metrics.Metric):
    def __init__(self, name="norm"):
        super(L2_norm, self).__init__(name=name)
        self.initial_weights = None
        self.l2_norm = self.add_weight(name="l2_norm", initializer="zeros")

    def reset_state(self):
        # Reset the l2_norm to zero
        self.l2_norm.assign(0.0)

    def result(self):
        # Return the current state of the l2_norm
        return self.l2_norm

    def set_weight_mean(self, weights):
        # Deep copy the initial weights
        self.initial_weights = [tf.identity(w) for w in weights]

    def update_state(self, weights):
        # Ensure that the weights have been initialized before updating state
        if self.initial_weights is None:
            raise ValueError("Call set_weight_mean() with the initial weights first.")

        # Calculate the squared L2 norm of the weight differences
        norm_change = tf.constant(0.0)
        for start_w, end_w in zip(self.initial_weights, weights):
            norm_change += tf.reduce_sum(tf.square(end_w - start_w))
        self.l2_norm.assign(norm_change)

    # def set_weight_mean(self, weights):
    #     # Deep copy the initial weights
    #     self.initial_weights = [np.copy(w.numpy()) for w in weights]
    #     self.reset_state()  # Reset the l2_norm to ensure starting from zero

    # def update_state(self, new_weights):
    #     # Ensure that the weights have been initialized before updating state
    #     if self.initial_weights is None:
    #         raise ValueError(
    #             "Call set_weight_mean() with the initial weights first."
    #         )

    #     # Calculate the squared L2 norm of the weight differences
    #     norm_change = 0.0
    #     for start_w, end_w in zip(self.initial_weights, new_weights):
    #         # Convert tensors to numpy arrays for calculation
    #         end_w_array = end_w.numpy()
    #         norm_change += np.sum(np.square(end_w_array - start_w))
    #     self.l2_norm = norm_change
