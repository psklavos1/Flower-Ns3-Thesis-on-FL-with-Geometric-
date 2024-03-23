# third party
import flwr as fl
from flwr.common import Scalar, NDArrays
import tensorflow as tf

# built-in
import os
from typing import (
    List,
    Tuple,
    Dict,
)

# local
from .callback import CustomCallback
from utils.dataset import get_dataset, generate_class_percentages
from utils.model import CustomModel

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
# print(tf.config.list_physical_devices("GPU"))


class FlexibleClient(fl.client.NumPyClient):
    """
    @extends flwr.client.NumPyClient
    A flower client that is used to federate the learning process.

    Methods:
        get_parameters(config): Return the model weights.
        fit(parameters, config): Train the local model and return metrics for aggregation.
        evalueate(parameters, config): Test the local model.
    """

    def __init__(
        self,
        num_clients=2,
        partition_id=0,
        ds_name="mnist",
        non_iid=False,
        bias_template=0,
    ):
        # class variables
        self.batch_pointer: int = 0  # keep track of processed batches during training
        self.local_trainset: tf.data.Dataset = None
        self.local_testset: Tuple(List, List) = None
        self.local_validationset: tf.data.Dataset = None
        self.model: CustomModel = None

        # Map dataset names to their respective functions
        dataset_model_map = {
            "mnist": ("mnist", CustomModel),
            "fashion_mnist": ("fashion_mnist", CustomModel),
            "cifar10": ("cifar10", CustomModel),
            "cifar100": ("cifar100", CustomModel),
        }

        self._setup_dataset_and_model(
            dataset_model_map,
            ds_name,
            num_clients,
            partition_id,
            non_iid,
            bias_template,
        )

    # ====================================================================================================#
    # * Federated Learning
    def get_parameters(self, config):
        """
        @override
        Implementation to return the local model weights.
        Used to initialize global model from random client if initial state not specified.
        For more information refer to the fl.client.NumPyClient documentation
        """
        return self.model.get_weights()

    def fit(self, parameters, config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        @override
        Implementation to train the local model.
        For more information refer to the fl.client.NumPyClient documentation
        """

        # 1. Update local model with global params
        self.model.set_weights(parameters)

        # 2. Round start intializations
        metrics = {}
        self.model.train_l2_norm.set_weight_mean(self.model.trainable_weights)
        self.model.resume_train()

        # 3. Decrypt config:
        num_epochs, batch_size, verbose = (
            config["epochs"],
            config["batch_size"],
            config["verbose"],
        )

        # 4. Train the local model
        (
            batches_in_epoch,
            self.batch_pointer,
            l2_norm,
            computation_time,
        ) = self.model.fit(
            self.local_trainset,
            epochs=num_epochs,
            batch_size=batch_size,
            batch_pointer=self.batch_pointer,
            verbose=verbose,
            validation_data=self.local_validationset,
            callbacks=[CustomCallback(self.model)],
        )

        # 5. Prepare values to return
        processed_samples = batches_in_epoch * batch_size
        metrics["l2_norm"] = l2_norm
        metrics["computation_time"] = computation_time
        return self.model.get_weights(), processed_samples, metrics

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        @override
        evaluate a model after training.
        For more information refer to the flwr.client.NumPyClient documentation
        """

        # 1. Update local model with global params
        self.model.set_weights(parameters)

        # 2. Decrypt config:
        verbose, batch_size = config["verbose"], config["batch_size"]

        # 3. Evaluate local model
        x_test, y_test = self.local_testset
        res = self.model.evaluate(
            x_test, y_test, batch_size=batch_size, verbose=verbose
        )

        # 4. Prepate values to return
        _, _, loss, accuracy = res
        metrics = {"accuracy": accuracy}
        return loss, len(x_test), metrics

    # ====================================================================================================#
    # * Utility Functions

    def _setup_dataset_and_model(
        self,
        ds_model_map: Dict[str, CustomModel],
        ds_name: str,
        num_clients: int,
        partition_id: int,
        non_iid: bool,
        bias_template: int,
    ):
        """
        Setup the dataset and training model in accordance with the dataset name.

        Parameters:
            ds_model_map (Dict[str,CustomModel]): map the ds_name to the respective model.
            ds_name (str): The dataset name.
            num_clients (int): The number of clients, same as the number of dataset partitions.
            partition_id (int): The index of the partition for this client.
            non_iid (bool): Whether there data are non_iid or not.
            bias_template (int): Indicate the level of bias. 0 -> random bias. 1-3 -> bias in increasing order.
        """
        if ds_name in ds_model_map:

            dataset_name, model_class = ds_model_map[ds_name]

            # Percentages generation
            class_percentages = None
            if non_iid == True:
                class_percentages = generate_class_percentages(
                    dataset_name, bias_template
                )

            (
                self.local_trainset,
                self.local_testset,
                self.local_validationset,
            ) = get_dataset(
                num_partitions=num_clients,
                partition_index=partition_id,
                ds_name=dataset_name,
                non_iid=non_iid,
                class_percentages=class_percentages,
            )
            self.model = model_class(ds_name=dataset_name)
            self.model.compile()
        else:
            raise ValueError(f"Invalid dataset name '{ds_name}'")
<<<<<<< Updated upstream

        self.model.compile()

    # * Federated Learning
    # Initialize the global model from a random client and it takes its weights
    def get_parameters(self, config):
        return self.model.get_weights()

    # send the global parameters and train model
    def fit(self, parameters, config):
        # Update local model with global params
        self.model.set_weights(parameters)

        # Decrypt config:
        num_epochs, batch_size, verbose = (
            config["epochs"],
            config["batch_size"],
            config["verbose"],
        )

        metrics = {}

        # Round start resets
        self.model.train_l2_norm.set_weight_mean(self.model.trainable_weights)
        self.model.resume_train()

        # Train the local model
        batches_in_epoch, self.batch_pointer, l2_norm = self.model.fit(
            self.local_trainset,
            epochs=num_epochs,
            batch_size=batch_size,
            batch_pointer=self.batch_pointer,
            verbose=verbose,
            validation_data=self.local_validationset,
            callbacks=[CustomCallback(self.model)],
        )
        processed_samples = batches_in_epoch * self.batch_pointer
        metrics["l2_norm"] = l2_norm
        return self.model.get_weights(), processed_samples, metrics

    def evaluate(self, parameters, config):
        # Decrypt config:
        verbose, batch_size = config["verbose"], config["batch_size"]
        # Set the weights to the ones sent by the global server
        self.model.set_weights(parameters)
        x_test, y_test = self.local_testset

        res = self.model.evaluate(
            x_test, y_test, batch_size=batch_size, verbose=verbose
        )

        _, _, loss, accuracy = res
        metrics = {"accuracy": accuracy}
        return loss, len(x_test), metrics
=======
>>>>>>> Stashed changes
