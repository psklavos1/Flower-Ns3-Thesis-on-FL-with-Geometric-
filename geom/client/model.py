# third party
import io
import math
import os
import tensorflow as tf
import grpc
import numpy as np
import tempfile


# built-in
import time

# local
from protos import metric_service_pb2_grpc
from utils.l2_norm import L2_norm


# Error Declaration
class ModelNotInitializedError(Exception):
    """Exception raised when the model is not initialized."""

    def __init__(
        self, message="Model not initialized. Please build the model before calling it."
    ):
        self.message = message
        super().__init__(self.message)


class CustomModel(tf.keras.Model):
    def __init__(self, ds_name):
        super(CustomModel, self).__init__()
        self.ds_name = ds_name  # Store the dataset name
        self.build_model(ds_name)  # Dynamically build the model

        # Common optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        # Initialize common metrics
        self.init_metrics()

        # Monitoring
        self.stop_training = False

        # Additional setup steps
        self.channel, self.stub = self.setup_conn()
        # In configure maybe use the predict step to init the model

        self.configure(self.ds_name)
        self.summary()
        self.size = self.get_pickle_size()

    # Build Functions for the models
    def build_model(self, ds_name):
        """Select and build the model based on the dataset name."""
        model_builders = {
            "mnist": self.build_mnist_model,
            "fashion_mnist": self.build_fashion_mnist_model,
            "cifar10": self.build_cifar10_model,
            "cifar100": self.build_cifar100_model,
        }

        if ds_name in model_builders:
            model_builders[ds_name]()  # Build the model
        else:
            raise ValueError(f"Unsupported dataset: {ds_name}")

    def build_mnist_model(self):
        """Build a simpler model suitable for MNIST."""
        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(28, 28, 1)
        )
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(64, activation="relu")
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")
        print("Model built for MNIST")

    def build_fashion_mnist_model(self):
        """Build a more complex model suitable for Fashion MNIST."""
        self.conv1 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(28, 28, 1)
        )
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
        self.maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")
        print("Model built for Fashion MNIST")

    def build_cifar10_model(self):
        """Build a model suitable for CIFAR-10."""
        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)
        )
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)

        self.conv3 = tf.keras.layers.Conv2D(
            64, (3, 3), padding="same", activation="relu"
        )
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.25)

        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(512, activation="relu")
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.d2 = tf.keras.layers.Dense(10, activation="softmax")
        print("Model built for CIFAR-10")

    def build_cifar100_model(self):
        self.conv1 = tf.keras.layers.Conv2D(
            64, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)
        )
        self.conv2 = tf.keras.layers.Conv2D(
            128, (3, 3), padding="same", activation="relu"
        )
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.4)

        self.conv3 = tf.keras.layers.Conv2D(
            256, (3, 3), padding="same", activation="relu"
        )
        self.conv4 = tf.keras.layers.Conv2D(
            256, (3, 3), padding="same", activation="relu"
        )
        self.maxpool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.4)

        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.d2 = tf.keras.layers.Dense(
            100, activation="softmax"
        )  # 100 classes in CIFAR-100
        print("Model built for CIFAR-100")

    def call(self, inputs, training=False):
        """Forward pass logic, selected based on the dataset."""
        if self.ds_name == "mnist":
            return self.forward_mnist(inputs)
        elif self.ds_name == "fashion_mnist":
            return self.forward_fashion_mnist(inputs)
        elif self.ds_name == "cifar10":
            return self.forward_cifar10(inputs)
        elif self.ds_name == "cifar100":
            return self.forward_cifar100(inputs, training)
        else:
            raise ValueError(f"Unsupported dataset: {self.ds_name}")

    # Forward functions for different datasets
    def forward_mnist(self, inputs):
        """Forward pass for the MNIST model."""
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    def forward_fashion_mnist(self, inputs):
        """Forward pass for the Fashion MNIST model."""
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    def forward_cifar10(self, inputs, training=False):
        """Forward pass for the CIFAR-10 model."""
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x, training=training)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x, training=training)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout3(x, training=training)
        return self.d2(x)

    def forward_cifar100(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x, training=training)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x, training=training)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout3(x, training=training)
        return self.d2(x)

    def init_metrics(self):
        """Initialize common metrics."""
        # Train Metrics
        self.train_loss_tracker = tf.keras.metrics.Mean(name="mean_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
        self.train_l2_norm = L2_norm()
        # Val Metrics
        self.val_loss_tracker = tf.keras.metrics.Mean(name="mean_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def configure(self, ds_name):
        """Additional configuration (placeholder method)."""
        if ds_name in ["mnist", "fashion_mnist"]:
            self.build(input_shape=(None, 28, 28, 1))
            self.summary()
            dummy_data = np.random.random(
                (1, 28, 28, 1)
            )  # Create one sample of dummy data
            _ = self.predict(dummy_data)
        elif ds_name in ["cifar10", "cifar100"]:
            self.build(input_shape=(None, 32, 32, 3))  # Adjust input shape for CIFAR-10
            dummy_data = np.random.random(
                (1, 32, 32, 3)
            )  # Create one sample of dummy CIFAR-10 data
            _ = self.predict(dummy_data)  # Use the dummy data to build the model

    def setup_conn(self, address="127.0.0.1:8091"):
        channel = grpc.insecure_channel(address)
        stub = metric_service_pb2_grpc.MetricServiceStub(channel)
        return channel, stub

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.train_loss_tracker,
            self.train_accuracy,
            self.val_loss_tracker,
            self.val_accuracy,
        ]

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self.loss_fn(y, logits)

        # Apply gradients
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update Metrics
        self.train_accuracy.update_state(y, logits)
        self.train_loss_tracker.update_state(loss)
        self.train_l2_norm.update_state(self.trainable_weights)
        return loss

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        # Update Metrics
        loss = self.loss_fn(y, y_pred)
        self.val_accuracy.update_state(y, y_pred)
        self.val_loss_tracker.update_state(loss)

        # print(f"mean_loss: {self.val_loss_tracker.result()}, accuracy: {self.val_accuracy.result()}")
        return {
            "mean_loss": self.val_loss_tracker.result(),
            "accuracy": self.val_accuracy.result(),
        }

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        batch_pointer=0,
        batch_size=32,
        epochs=1,
        verbose=1,
        validation_data=None,
        callbacks=[],
    ):
        start_fit = time.time()
        batches_in_epoch = 0
        # Convert data to a tf.data.Dataset
        train_dataset = train_dataset.batch(batch_size)

        for epoch in range(epochs):
            epoch_start_time = time.time()
            elapsed_time = 0
            print(f"========================== Epoch Start ==========================")

            # Callback for the start of the epoch
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            l2_norm_val = 0
            train_dataset = train_dataset.skip(batch_pointer)

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                # Stop check
                if self.stop_training:
                    break

                # callback for start of batch
                for callback in callbacks:
                    callback.on_train_batch_begin(step)

                loss_val = self.train_step(x_batch_train, y_batch_train)
                batches_in_epoch += 1
                batch_pointer += 1

                l2_norm_val = float(self.train_l2_norm.result())

                # callback for end of batch
                for callback in callbacks:
                    callback.on_train_batch_end(step)

                # Verbose = 0: Silent, 1: per 200 batches, 2: per 10 batches
                if verbose == 1:
                    if step % 200 == 0:
                        elapsed_time = time.time() - epoch_start_time
                        self.print_results(
                            batches_in_epoch, batch_pointer, loss_val, elapsed_time
                        )

                elif verbose == 2:
                    # Log every 10 batches.
                    if step % 10 == 0:
                        elapsed_time = time.time() - epoch_start_time
                        self.print_results(
                            batches_in_epoch, batch_pointer, loss_val, elapsed_time
                        )

                else:
                    if batches_in_epoch + step == 0:
                        print(
                            "Verbose not set. Awaiting Results\nThis message will be printed once."
                        )

            # Display metrics at the end of each epoch.
            print("Epoch Results:")
            self.print_results(batches_in_epoch, batch_pointer, loss_val, elapsed_time)
            # Reset training metrics at the end of each epoch
            self.reset_train_metrics()

            # Callback for end of Epoch
            for callback in callbacks:
                callback.on_epoch_end(epoch)

            computation_time = time.time() - start_fit
            if validation_data is not None:
                print(f"========================= Validation =========================")
                # Run a validation loop at the end of each epoch.
                for x_batch_val, y_batch_val in validation_data:
                    self.test_step(x_batch_val, y_batch_val)

                val_acc = self.val_accuracy.result()
                val_loss = self.val_loss_tracker.result()
                self.reset_val_metrics()
                print(
                    f"Validation Avg Results\t Loss:{float(val_loss)}\t Accuracy: {float(val_acc):.4f}"
                )

        return batches_in_epoch, batch_pointer, l2_norm_val, computation_time

    def reset_train_metrics(self):
        self.train_loss_tracker.reset_states()
        self.train_accuracy.reset_states()
        self.train_l2_norm.reset_state()

    def reset_val_metrics(self):
        self.val_loss_tracker.reset_states()
        self.val_accuracy.reset_states()

    def resume_train(self):
        self.stop_training = False

    def print_results(self, batches_in_epoch, batch_pointer, loss_val, elapsed_time):
        print(
            f"(In epoch/Overall): ({batches_in_epoch}/{batch_pointer})| inst_loss: {float(loss_val):.4f} - Mean loss: {float(self.train_loss_tracker.result()):.4f} - acc: {float(self.train_accuracy.result()):.4f} - norm: {float(self.train_l2_norm.result()):.2f} - {elapsed_time:.2f}s"
        )

    def get_pickle_size(self):
        import pickle
        import sys

        # Assuming `model` is your model object
        serialized_model = pickle.dumps(self)
        size_kbs = sys.getsizeof(serialized_model) / 1024
        print(f"Size of serialized model: {size_kbs} kbytes")
        return size_kbs

    def get_size(self):
        weights = self.get_weights()

        # Serialize weights to a byte stream
        buffer = io.BytesIO()
        np.save(buffer, weights, allow_pickle=True)

        # Get the size of the serialized weights
        transmission_size_bytes = buffer.tell()
        transmission_size_kilobytes = transmission_size_bytes / 1024
        print(f"transmission_size_bytes {transmission_size_kilobytes}")
        return transmission_size_kilobytes
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     # Save the model to the temporary directory
        #     tf.saved_model.save(self, temp_dir)

        #     # Calculate the size of the saved model
        #     total_size = 0
        #     for dirpath, dirnames, filenames in os.walk(temp_dir):
        #         for f in filenames:
        #             fp = os.path.join(dirpath, f)
        #             total_size += os.path.getsize(fp)
        #     print(f"Size of saved model: {total_size/1024}")
        #     return total_size / 1024  # Size in kilobytes
