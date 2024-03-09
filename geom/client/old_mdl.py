# third party
import tensorflow as tf
import grpc

# built-in
import time
import io

# local
from protos import metric_service_pb2_grpc
from utils.l2_norm import L2_norm


class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel, self.stub = self.setup_conn()
        # Loss fn
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # Train Metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="mean_loss")
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.l2_norm = L2_norm()
        # Monitoring
        self.processed_examples = 0
        self.stop_training = False

    def __del__(self):
        if hasattr(self, "channel") and self.channel is not None:
            self.channel.close()

    def resume_train(self):
        self.stop_training = False

    def setup_conn(self, address="127.0.0.1:8091"):
        channel = grpc.insecure_channel(address)
        stub = metric_service_pb2_grpc.MetricServiceStub(channel)
        return channel, stub

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.loss_fn(y, y_pred)  # Compute loss

        # Compute & apply gradients
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(y, y_pred)
        self.l2_norm.update_state(self.trainable_weights)
        return loss

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.loss_fn(y, y_pred)
        # Update evaluation metrics
        self.accuracy.update_state(y, y_pred)
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.accuracy.result(),
        }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.accuracy]

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs=1,
        batch_size=32,
        verbose=1,
        batch_pointer=0,
        callbacks=[],
    ):
        # Convert data to a tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
            batch_size
        )
        num_batches = len(x_train) // batch_size

        for epoch in range(epochs):
            # Epoch loop
            epoch_start_time = time.time()
            print(f"================ Epoch {epoch + 1}/{epochs} ================")

            # Callback for the start of the epoch
            for callback in callbacks:
                callback.on_epoch_begin(epoch)
            l2_norm = 0
            # Batch Loop
            for step, (x_batch, y_batch) in enumerate(dataset):
                if step < batch_pointer:
                    continue

                batch_start_time = time.time()
                # Stop check
                if self.stop_training:
                    self.reset_metrics()
                    return self.processed_examples, l2_norm

                # Train
                for callback in callbacks:
                    callback.on_train_batch_begin(step)

                loss = self.train_step(x_batch, y_batch)
                self.processed_examples += len(x_batch)
                l2_norm = float(self.l2_norm.result())

                for callback in callbacks:
                    callback.on_train_batch_end(step)

                # Verbose printing to mimic TensorFlow's default fit method
                if verbose:
                    elapsed_time = time.time() - epoch_start_time
                    print(
                        f"{step + 1}/{num_batches+1} - inst_loss: {float(loss):.4f} - Mean loss: {float(self.loss_tracker.result()):.4f} - acc: {float(self.accuracy.result()):.4f} - norm: {float(self.l2_norm.result()):.2f} - {elapsed_time:.2f}s"
                    )
                else:
                    if step == 0:
                        print(
                            "Verbose not set. Awaiting Results\nThis message will be printed once."
                        )

            # End of Epoch
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch Duration: {epoch_duration}s")

            self.reset_metrics()
            for callback in callbacks:
                callback.on_epoch_end(epoch)

            # Validation Loop if needed
            if x_val is not None and y_val is not None:
                val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(
                    batch_size
                )

                for x_batch_val, y_batch_val in val_dataset:
                    self.test_step(x_batch_val, y_batch_val)

                val_acc = self.accuracy.result()
                val_loss = self.loss_tracker.result()
                self.reset_metrics()
                print(
                    f"Validation Avg Results\t Loss:{float(val_loss)}\t Accuracy: {float(val_acc):.4f}"
                )
            return self.processed_examples, l2_norm

    def reset_metrics(self):
        self.loss_tracker.reset_states()
        self.accuracy.reset_states()
        self.l2_norm.reset_state()

    def get_size(self) -> float:
        # Serialize the model to a byte stream
        with io.BytesIO() as byte_stream:
            tf.keras.models.save_model(self, byte_stream, save_format="h5")
            size = byte_stream.tell()  # Get the size of the byte stream
        # Convert bytes to kilobytes
        size = size / 1024
        print(f"Model size: {size}")
        return size
