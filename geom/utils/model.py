# third party
import tensorflow as tf
import keras
from keras.models import Model
from keras import Sequential
from keras.applications import ResNet50, VGG16
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Lambda
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy, Mean

import grpc
import pickle
import math

# built-in
import io
import numpy as np
import sys
import time
import logging

# local
from protos import metric_service_pb2_grpc
from utils.l2_norm import L2_norm
from utils.history_callback import CustomHistory

# from client.callback import CustomHistory


# Error Declaration
class ModelNotInitializedError(Exception):
    """Exception raised when the model is not initialized."""

    def __init__(
        self, message="Model not initialized. Please build the model before calling it."
    ):
        self.message = message
        super().__init__(self.message)


class CustomModel(keras.Model):
    """
    A custom Keras model class tailored for specific datasets (MNIST, Fashion MNIST, CIFAR-10, CIFAR-100).

    This class extends keras.Model to provide dataset-specific model architectures and includes
    methods for training, evaluation, and utilities for managing metrics and gRPC communication.

    Attributes:
        ds_name (str): The name of the dataset for which the model is built.
        optimizer (keras.optimizers.Optimizer): Optimizer for model training.
        loss_fn (keras.losses): Loss function for model training.
        train_loss_tracker (keras.metrics.Mean): Tracks mean loss during training.
        train_accuracy (keras.metrics.SparseCategoricalAccuracy): Tracks accuracy during training.
        train_l2_norm (L2_norm): Custom metric to track the L2 norm of model weights changes.
        val_loss_tracker (keras.metrics.Mean): Tracks mean loss during validation.
        val_accuracy (keras.metrics.SparseCategoricalAccuracy): Tracks accuracy during validation.
        stop_training (bool): Flag to stop training early.
        channel (grpc.Channel): gRPC channel for communication.
        stub (Stub): gRPC stub for making requests.

    Methods:
        build_model(ds_name): Dynamically builds the model based on the dataset name.
        call(inputs, training=False): Forward pass logic for different datasets.
        train_step(x, y): Performs a single step of training.
        test_step(data): Performs a single step of evaluation.
        fit(train_dataset, ...): Custom training loop.
        get_pickle_size(): Returns the size of the serialized model.
        get_size(): Calculates the size of the model weights for transmission.
    """

    # ===============================================================================================================
    # * Initialization and Configuration
    def __init__(self, ds_name, ann_name):
        super(CustomModel, self).__init__()
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        self.ds_name = ds_name
        self.ann_name = ann_name
        self.supported_datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

        self.model = self.build_model(ds_name, ann_name)  # Dynamically build the model

        # Common optimizer and loss function
        self.optimizer = Adam()
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=False)

        # Initialize common metrics
        self._init_metrics()

        # Monitoring
        self.stop_training = False

        # Additional setup steps
        self.channel, self.stub = self._setup_conn()
        # In configure maybe use the predict step to init the model
        self._configure(self.ds_name)
        self.model.summary()
        self.size = self.get_pickle_size()

    def _configure(self, ds_name):
        """
        Performs additional configuration for the model based on the dataset.

        Args:
            ds_name (str): Name of the dataset for configuration adjustments.
        """
        if ds_name in ["mnist", "fashion_mnist"]:
            self.build(input_shape=(None, 28, 28, 1))
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

    def build_model(self, ds_name, ann_name):
        """
        Builds the model architecture dynamically based on the dataset name.

        Args:
            ds_name (str): Name of the dataset.

        Returns:
            The model.

        Raises:
            ValueError: If ds_name is not supported.
        """

        num_classes = 100 if ds_name == "cifar100" else 10
        supported_model_builders = {
            "lenet": self._build_lenet,
            "alexnet": self._build_alexnet,
            "resnet": self._build_resnet,
            "vgg": self._build_vgg,
        }

        input_shape_per_dataset = {
            "mnist": (28, 28, 1),
            "fashion_mnist": (28, 28, 1),
            "cifar10": (32, 32, 3),
            "cifar100": (32, 32, 3),
        }

        default_model_builders = {
            "mnist": self._build_mnist_model,
            "fashion_mnist": self._build_fashion_mnist_model,
            "cifar10": self._build_cifar10_model,
            "cifar100": self._build_cifar100_model,
        }
        if ds_name in self.supported_datasets:
            if ann_name in supported_model_builders:
                return supported_model_builders[ann_name](
                    input_shape=input_shape_per_dataset[ds_name],
                    num_classes=num_classes,
                )
            else:
                return default_model_builders[ds_name]()
        else:
            raise ValueError(f"Unsupported dataset: {ds_name}")

    # ===============================================================================================================
    # * Ready to use models
    def _build_lenet(self, input_shape, num_classes):
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(6, (5, 5), activation="relu", padding="same"),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(16, (5, 5), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(120, activation="relu"),
                Dense(84, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ]
        )
        self.logger.info("LeNet model built")
        return model

    def _build_resnet(self, input_shape, num_classes):
        model = ResNet50(weights=None, input_shape=input_shape, classes=num_classes)
        self.logger.info("ResNet50 model built")

        return model

    def _build_vgg(self, input_shape, num_classes):
        # Convert grayscale images to RGB by repeating the single channel three times
        if input_shape[2] == 1:
            inputs = Input(shape=input_shape)
            x = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)
        else:
            inputs = Input(shape=input_shape)
            x = inputs

        # Use VGG16 with the modified input
        base_model = VGG16(weights=None, include_top=False, input_tensor=x)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs, outputs)
        self.logger.info("VGG16 model built")
        return model

    # Adjusted to be able to fit smaller images. Under consideration of wherther to use
    def _build_alexnet(self, input_shape, num_classes):
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Conv2D(192, (3, 3), padding="same", activation="relu"),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Conv2D(384, (3, 3), padding="same", activation="relu"),
                Conv2D(256, (3, 3), padding="same", activation="relu"),
                Conv2D(256, (3, 3), padding="same", activation="relu"),
                MaxPooling2D(pool_size=(2, 2), strides=2),
                Flatten(),
                Dense(4096, activation="relu"),
                Dropout(0.5),
                Dense(4096, activation="relu"),
                Dropout(0.5),
                Dense(num_classes, activation="softmax"),
            ]
        )
        self.logger.info("AlexNet model built")
        return model

    # * Model Building by Dataset
    def _build_mnist_model(self):
        """Build a simpler model suitable for MNIST."""
        model = Sequential(
            [
                Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(10, activation="softmax"),
            ]
        )
        self.logger.info("Default Model built for MNIST")
        return model

    def _build_fashion_mnist_model(self):
        """Build a more complex model suitable for Fashion MNIST."""
        model = Sequential(
            [
                Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(10, activation="softmax"),
            ]
        )
        self.logger.info("Default Model built for Fashion MNIST")
        return model

    def _build_cifar10_model(self):
        """Build a model suitable for CIFAR-10."""
        model = Sequential(
            [
                Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=(32, 32, 3),
                ),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                Conv2D(64, (3, 3), padding="same", activation="relu"),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(10, activation="softmax"),
            ]
        )
        self.logger.info("Default Model built for CIFAR-10")
        return model

    def _build_cifar100_model(self):
        model = Sequential(
            [
                Conv2D(
                    64,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=(32, 32, 3),
                ),
                Conv2D(128, (3, 3), padding="same", activation="relu"),
                MaxPooling2D((2, 2)),
                Dropout(0.4),
                Conv2D(256, (3, 3), padding="same", activation="relu"),
                Conv2D(256, (3, 3), padding="same", activation="relu"),
                MaxPooling2D((2, 2)),
                Dropout(0.4),
                Flatten(),
                Dense(1024, activation="relu"),
                Dropout(0.5),
                Dense(100, activation="softmax"),
            ]
        )
        self.logger.info("Default Model built for CIFAR-100")
        return model

    # ===============================================================================================================
    # * Forward Pass Implementations
    def call(self, inputs, training=False):
        """
        Forward pass for the model, selecting the appropriate logic based on the dataset.

        Args:
            inputs: Input data.
            training (bool): Whether the forward pass is for training or inference.

        Returns:
            The model's output predictions.

        Raises:
            ValueError: If an unsupported dataset name is provided.
        """
        if self.ds_name in self.supported_datasets:
            return self._forward_pass(inputs, training=training)
        else:
            raise ValueError(f"Unsupported dataset: {self.ds_name}")

    def _forward_pass(self, inputs, training=False):
        """
        Forward pass logic

        Args:
            inputs: Input data for the model.

        Returns:
            Output predictions for the model.
        """
        x = self.model(inputs, training=training)
        return x

    # ===============================================================================================================
    # * Training and Evaluation Methods
    @tf.function
    def train_step(self, x, y):
        """
        Performs a single training step.

        Args:
            x: Input data.
            y: Target data.

        Returns:
            The scalar loss value resulting from the training step.
        """
        with tf.GradientTape() as tape:
            logits = self(x, training=True)
            loss = self.loss_fn(y, logits)

        # Apply gradients
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update Metrics
        self.train_accuracy.update_state(y, logits)
        self.train_loss_tracker.update_state(loss)
        self.train_l2_norm.update_state(self.trainable_variables)
        return loss

    @tf.function
    def test_step(self, data):
        """
        Performs a single evaluation step.

        Args:
            data: Tuple of (input data, target data).

        Returns:
            A dictionary with keys 'mean_loss' and 'accuracy' for the evaluation metrics.
        """
        x, y = data
        y_pred = self(x, training=False)
        # Update Metrics
        loss = self.loss_fn(y, y_pred)
        self.val_accuracy.update_state(y, y_pred)
        self.val_loss_tracker.update_state(loss)

        return {
            "val_loss": self.val_loss_tracker.result(),
            "val_accuracy": self.val_accuracy.result(),
        }

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        batch_pointer=0,
        batch_size=32,
        epochs=1,
        verbose=1,
        valid_dataset=None,
        round=1,
        callbacks=[],
    ) -> keras.callbacks.History:
        """
        Custom training loop for the model.

        Args:
            train_dataset (tf.data.Dataset): Dataset for training.
            batch_pointer (int): Batch pointer to resume training from.
            batch_size (int): Size of batches for training.
            epochs (int): Number of epochs to train for.
            verbose (int): Verbosity mode.
            valid_dataset (tf.data.Dataset, optional): Dataset for validation.
            callbacks (list): List of callbacks for training.

        Returns:
            CustomHistory object containing training history.
        """
        total_computation_time = tf.Variable(
            0, dtype=tf.float64
        )  # cumulative for epoch
        total_rtc_check_time = tf.Variable(0, dtype=tf.float64)  # cumulative for epoch

        batches_in_epoch = 0
        # batch dataset
        start_fit = tf.timestamp()
        train_dataset = train_dataset.batch(batch_size)
        if valid_dataset is not None:
            valid_dataset = valid_dataset.batch(2 * batch_size)

        custom_history = CustomHistory()
        callbacks.append(custom_history)

        # Initialize callbacks
        for callback in callbacks:
            callback.set_model(self)
            callback.on_train_begin()

        for epoch in range(epochs):
            epoch_start_time = tf.timestamp()
            elapsed_time = 0
            self.logger.info(
                "========================== Epoch Start =========================="
            )

            # Callback for the start of the epoch
            for callback in callbacks:
                callback.on_epoch_begin(epoch)

            l2_norm_value = 0.0
            train_dataset = train_dataset.skip(batch_pointer)
            loss_val = 0.0
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                batch_start_time = tf.timestamp()

                # Stop check
                if self.stop_training:
                    break

                # Callback for start of batch
                for callback in callbacks:
                    callback.on_train_batch_begin(step)

                ################################################################
                comp_start_time = tf.timestamp()
                loss_val = self.train_step(x_batch_train, y_batch_train)
                # print(f"Computation Time: {(tf.timestamp()-comp_start_time).numpy()}")
                computation_dur = (tf.timestamp() - comp_start_time).numpy()
                ################################################################
                batches_in_epoch += 1
                batch_pointer += 1

                l2_norm_value = float(self.train_l2_norm.result())

                batch_logs = {
                    "round": round,
                    "step_no": batch_pointer,
                    "loss": float(loss_val),
                    "accuracy": float(self.train_accuracy.result()),
                    "l2_norm": l2_norm_value,
                    "computation_time": computation_dur,
                    "batch_time": (tf.timestamp() - batch_start_time).numpy(),
                }  # Callback for end of batch

                for callback in callbacks:
                    callback.on_train_batch_end(step, batch_logs)

                # print(f"Communication Time: {batch_logs.get('rtc_check_time', 'Not available')}")

                total_computation_time.assign_add(computation_dur)
                total_rtc_check_time.assign_add(batch_logs.get("rtc_check_time", 0))

                # Verbose = 0: Silent, 1: per 100 batches, 2: per 10 batches
                if verbose == 1:
                    if step % 100 == 0:
                        elapsed_time = (tf.timestamp() - epoch_start_time).numpy()
                        self._print_results(
                            batches_in_epoch, batch_pointer, elapsed_time, loss_val
                        )

                elif verbose == 2:
                    # Log every 10 batches.
                    if step % 10 == 0:
                        elapsed_time = (tf.timestamp() - epoch_start_time).numpy()
                        self._print_results(
                            batches_in_epoch, batch_pointer, elapsed_time, loss_val
                        )

                else:
                    if batches_in_epoch + step == 0:
                        self.logger.info(
                            "Verbose not set. Awaiting Results\nThis message will be printed once."
                        )

                # print(f"Total batch time: {(tf.timestamp() - batch_start_time).numpy()}")

            # Display metrics at the end of each epoch.
            total_fit_time = tf.timestamp() - start_fit
            self.logger.info("Epoch Results:")
            self._print_results(
                batches_in_epoch, batch_pointer, loss_val, total_fit_time.numpy()
            )
            # Reset training metrics at the end of each epoch
            self._reset_train_metrics()

            epoch_logs = {
                "round": round,
                "epoch_steps": batches_in_epoch,
                "total_steps": batch_pointer,
                "l2_norm": l2_norm_value,
                "total_computation_time": total_computation_time.numpy(),
                "total_rtc_check_time": total_rtc_check_time.numpy(),
                "total_duration": total_fit_time.numpy(),
            }

            # Callback for end of Epoch
            for callback in callbacks:
                callback.on_epoch_end(epoch, epoch_logs)

            # Collect training metrics
            if valid_dataset is not None:
                self.logger.info(
                    "\n================================================== Train Validation =================================================="
                )
                # Run a validation loop at the end of each epoch.
                for step, val_pair in enumerate(valid_dataset):
                    val_results = self.test_step(val_pair)
                    if verbose != 0:
                        self.logger.info(
                            f"Validation Batch {step+1} - mean_loss: {val_results['val_loss']:.4f}, accuracy: {val_results['val_accuracy']:.4f}"
                        )

                val_acc = self.val_accuracy.result()
                val_loss = self.val_loss_tracker.result()
                self._reset_val_metrics()
                self.logger.info(
                    f"Validation Avg Results\t Loss:{float(val_loss):.4f}\t Accuracy: {float(val_acc):.4f}"
                )
                self.logger.info(
                    "==============================================================\n"
                )
                # Collect validation metrics
                epoch_logs["val_loss"] = float(val_loss)
                epoch_logs["val_accuracy"] = float(val_acc)

        # Ensure all callbacks complete their end-of-training routines
        for callback in callbacks:
            callback.on_train_end()

        # Return the custom history callback
        custom_history = next(cb for cb in callbacks if isinstance(cb, CustomHistory))
        return custom_history

    def evaluate(self, test_dataset, batch_size=64, verbose=1):
        """
        Custom evaluate method using the custom test_step function.

        Args:
            test_dataset: The dataset for evaluation.
            batch_size: The size of the batches.
            verbose: Verbosity mode.

        Returns:
            The evaluation metrics as a dictionary.
        """
        # Batch the test dataset
        test_dataset = test_dataset.batch(batch_size)
        step = 0
        for step, batch_data in enumerate(test_dataset):
            step_result = self.test_step(batch_data)
            if verbose:
                self.logger.info(
                    f"Step {step+1}/{len(test_dataset)} - val_loss: {step_result['val_loss'].numpy():.4f} - val_accuracy: {step_result['val_accuracy'].numpy():.4f}"
                )

        # Get final metrics
        final_metrics = {
            "val_loss": self.val_loss_tracker.result().numpy(),
            "val_accuracy": self.val_accuracy.result().numpy(),
        }

        self.logger.info(
            f"Evaluation Results - Val Loss: {final_metrics['val_loss']:.4f}, Val Accuracy: {final_metrics['val_accuracy']:.4f} in {step+1} steps"
        )
        # Reset metrics for next evaluation
        self.reset_metrics()

        return final_metrics

    # ===============================================================================================================
    # * API for seamless underlaying model operation by delegating
    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, parameters):
        self.model.set_weights(parameters)

    # def compile(self, *args, **kwargs):
    #     return self.model.compile(*args, **kwargs)

    # ===============================================================================================================
    # * Metric and State Management
    def _init_metrics(self):
        """
        Initializes common training and validation metrics.
        """
        # Train Metrics
        self.train_loss_tracker = Mean(name="train_loss")
        self.train_accuracy = SparseCategoricalAccuracy(name="train_accuracy")
        self.train_l2_norm = L2_norm()

        # Val Metrics
        self.val_loss_tracker = Mean(name="val_loss")
        self.val_accuracy = SparseCategoricalAccuracy(name="val_accuracy")

    def _reset_train_metrics(self):
        """
        Resets the training metrics to their initial state.
        """

        self.train_loss_tracker.reset_states()
        self.train_accuracy.reset_states()
        self.train_l2_norm.reset_state()

    def _reset_val_metrics(self):
        """
        Resets the validation metrics to their initial state.
        """
        self.val_loss_tracker.reset_states()
        self.val_accuracy.reset_states()

    def resume_train(self):
        """
        Allows the resumption of training by resetting the stop_training flag.
        """
        self.stop_training = False

    # ===============================================================================================================
    # * Utility and Communication Methods

    def _setup_conn(self, address="127.0.0.1:8091"):
        """
        Sets up a gRPC channel and stub for communication.

        Args:
            address (str): The address of the gRPC server.

        Returns:
            A tuple of (grpc.Channel, grpc Stub) for communication.
        """
        channel = grpc.insecure_channel(address)
        stub = metric_service_pb2_grpc.MetricServiceStub(channel)
        return channel, stub

    def _print_results(self, batches_in_epoch, batch_pointer, loss_val, elapsed_time):
        """
        Utility method for printing training results during training.

        Args:
            batches_in_epoch (int): Number of batches processed in the current epoch.
            batch_pointer (int): Global batch pointer across epochs.
            loss_val (float): Instantaneous loss value.
            elapsed_time (float): Time elapsed since the start of the epoch.
        """
        self.logger.info(
            f"(In epoch/Overall): ({batches_in_epoch}/{batch_pointer})| inst_loss: {float(loss_val):.4f} - Mean loss: {float(self.train_loss_tracker.result()):.4f} - acc: {float(self.train_accuracy.result()):.4f} - norm: {float(self.train_l2_norm.result()):.2f} - {elapsed_time:.2f}s"
        )

    def get_pickle_size(self):
        """
        Calculates the size of the model when serialized with pickle.

        Returns:
            Size of the serialized model in kilobytes.
        """

        # Assuming `model` is your model object
        serialized_model = pickle.dumps(self)
        size_kbs = sys.getsizeof(serialized_model) / 1024
        self.logger.info(f"Size of serialized model: {size_kbs} kbytes")
        return size_kbs

    def get_size(self):
        """
        Determines the size of the model's weights for transmission over networks.

        Returns:
            The size of the serialized weights in kilobytes.
        """
        weights = self.get_weights()

        # Serialize weights to a byte stream
        buffer = io.BytesIO()
        np.save(buffer, weights, allow_pickle=True)

        # Get the size of the serialized weights
        transmission_size_bytes = buffer.tell()
        transmission_size_kilobytes = transmission_size_bytes / 1024
        self.logger.info(f"transmission_size_bytes {transmission_size_kilobytes}")
        return transmission_size_kilobytes

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
