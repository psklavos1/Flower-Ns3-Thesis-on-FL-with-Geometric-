# third party
import tensorflow as tf

# To create graph functions for efficiency
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


class L2_norm(tf.keras.metrics.Metric):
    """
    Custom metric to track the L2 norm of weight changes during training.

    Extends `tf.keras.metrics.Metric` to calculate the L2 norm difference between the initial weights and the current
    weights of a model during a training round. This metric helps in monitoring the magnitude of weight updates.

    Attributes:
        initial_weights: Stores the initial model weights at the start of a training round.
        l2_norm: The calculated L2 norm of the weight differences.

    Methods:
        reset_state: Resets the L2 norm calculations to zero.
        result: Returns the current L2 norm value.
        update_state: Updates the L2 norm based on the current weights versus initial weights.
        set_weight_mean: Sets the initial weights to the provided weights at the start of a training round.
    """

    def __init__(self, name="norm"):
        super(L2_norm, self).__init__(name=name)
        self.initial_weights = None
        self.l2_norm = self.add_weight(name="l2_norm", initializer="zeros")

    def reset_state(self):
        """
        Reset the L2 norm calculation to 0.0, preparing for a new round of calculation.
        """
        self.l2_norm.assign(0.0)

    def result(self):
        """
        Return the current value of the L2 norm calculation.

        Returns:
            tf.Tensor: The current L2 norm.
        """
        return self.l2_norm

    def update_state(self, weights):
        """
        Update the L2 norm calculation with the current weights of the model.

        Args:
            weights (List[tf.Tensor]): The current weights of the model.

        Raises:
            ValueError: If initial weights have not been set before calling this method.
        """
        if self.initial_weights is None:
            raise ValueError("Call set_weight_mean() with the initial weights first.")

        # Calculate the squared L2 norm of the weight differences
        norm_change = tf.constant(0.0)
        for start_w, end_w in zip(self.initial_weights, weights):
            norm_change += tf.reduce_sum(tf.square(end_w - start_w))
        self.l2_norm.assign(norm_change)

    def set_weight_mean(self, weights):
        """
        Set the initial weights at the start of a training round.

        Args:
            weights (List[tf.Tensor]): The initial weights of the model.
        """
        self.initial_weights = [tf.identity(w) for w in weights]
