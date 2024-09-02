# third party
import flwr as fl
from flwr.common import Scalar, NDArrays
import tensorflow as tf

# built-in
import os
import json
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
        ann_name=None,
        non_iid=False,
        bias_template=0,
        val_ratio=0.1,
        train_val_ratio=0.0,
        keep_log=False,
    ):
        self.keep_log = keep_log
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        # class variables
        self.batch_pointer: int = 0  # keep track of processed batches during training

        self.local_trainset: tf.data.Dataset = None
        self.local_train_val_set: tf.data.Dataset = None
        self.local_validationset: tf.data.Dataset = None
        self.model: CustomModel = None
        self.round_result_tracker = {"batch": [], "epoch": {}}

        # Supported datasets
        supported_datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

        self._setup_dataset_and_model(
            supported_datasets,
            ds_name,
            ann_name,
            num_clients,
            partition_id,
            non_iid,
            bias_template,
            val_ratio=val_ratio,
            train_val_ratio=train_val_ratio,
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
        self.round_result_tracker = {"batch": [], "epoch": {}}

        # 1. Update local model with global params
        self.model.set_weights(parameters)

        # 2. Round start initializations
        metrics = {}
        self.model.train_l2_norm.set_global_weight_estimate(
            self.model.trainable_weights
        )
        self.model.resume_train()

        # 3. Decrypt config:
        num_epochs, batch_size, verbose, round_num = (
            config["epochs"],
            config["batch_size"],
            config["verbose"],
            config["round"],
        )

        # 4. Train the local model
        history_cb = self.model.fit(
            self.local_trainset,
            epochs=num_epochs,
            batch_size=batch_size,
            batch_pointer=self.batch_pointer,
            verbose=verbose,
            valid_dataset=self.local_train_val_set,
            round=round_num,
            callbacks=[CustomCallback(self.model)],
        )

        # 5. Extract round resutls
        if self.keep_log:
            self._extract_round_results(history_cb)

        # 6. Prepare values to return
        processed_samples = (
            self._get_from_history(history_cb, "epoch_steps") * batch_size
        )
        self.batch_pointer = self._get_from_history(history_cb, "total_steps")

        metrics = {
            "l2_norm": self._get_from_history(history_cb, "l2_norm"),
            "computation_time": self._get_from_history(
                history_cb, "total_computation_time"
            ),
            "rtc_check_time": self._get_from_history(
                history_cb, "total_rtc_check_time"
            ),
            "fit_duration": self._get_from_history(history_cb, "total_duration"),
        }

        return self.model.get_weights(), processed_samples, metrics

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        @override
        Evaluate a model after training.
        For more information refer to the flwr.client.NumPyClient documentation.
        """
        # 1. Update local model with global params
        self.model.set_weights(parameters)

        # 2. Decrypt config:
        batch_size, verbose, round = (
            config["batch_size"],
            config["verbose"],
            config["round"],
        )

        verbose, batch_size = config["verbose"], config["batch_size"]
        # 3. Evaluate model
        res = self.model.evaluate(
            self.local_validationset,
            batch_size=batch_size,
            verbose=verbose,
        )

        # This part comes last so it returns once the data to be stored.
        if self.keep_log:
            self.round_result_tracker["epoch"].update(
                {
                    "val_loss": float(res["val_loss"]),
                    "val_accuracy": float(res["val_accuracy"]),
                }
            )

        # 4. Prepare values to return
        metrics = {
            "accuracy": float(res["val_accuracy"]),
            "epoch_data": json.dumps(
                self.round_result_tracker["epoch"]
            ),  # Serialize epoch data
            "batch_data": json.dumps(self.round_result_tracker["batch"]),  # Serialize
        }

        return (
            float(res["val_loss"]),
            int(tf.data.experimental.cardinality(self.local_validationset).numpy()),
            metrics,
        )

    # ====================================================================================================#
    # * Utility Functions
    def _get_from_history(self, history_cb, field: str):
        return history_cb.history[field][0]

    def _extract_round_results(self, history_cb):
        # Extract step-wise results
        batch_results = history_cb.batch_metrics

        # 6. Extract epoch-wise results (only the first element since epochs=1
        epoch_results = {
            "round": self._get_from_history(history_cb, "round"),
            "epoch_steps": self._get_from_history(history_cb, "epoch_steps"),
            "total_steps": self._get_from_history(history_cb, "total_steps"),
            "l2_norm": self._get_from_history(history_cb, "l2_norm"),
            "total_computation_time": self._get_from_history(
                history_cb, "total_computation_time"
            ),
            "total_rtc_check_time": self._get_from_history(
                history_cb, "total_rtc_check_time"
            ),
            "total_duration": self._get_from_history(history_cb, "total_duration"),
        }
        self.round_result_tracker["epoch"] = epoch_results
        self.round_result_tracker["batch"] = batch_results

    def _setup_dataset_and_model(
        self,
        supported_datasets: List[str],
        ds_name: str,
        ann_name: str,
        num_clients: int,
        partition_id: int,
        non_iid: bool,
        bias_template: int,
        val_ratio: float,
        train_val_ratio: float,
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
        if ds_name in supported_datasets:

            # Percentages generation
            class_percentages = None
            if non_iid == True:
                class_percentages = generate_class_percentages(ds_name, bias_template)

            (
                self.local_trainset,
                self.local_validationset,
                self.local_train_val_set,
            ) = get_dataset(
                num_partitions=num_clients,
                partition_index=partition_id,
                ds_name=ds_name,
                validation_split=val_ratio,
                train_val_split=train_val_ratio,
                non_iid=non_iid,
                class_percentages=class_percentages,
            )

            self.model = CustomModel(ds_name=ds_name, ann_name=ann_name)
            self.model.compile()
        else:
            raise ValueError(f"Invalid dataset name '{ds_name}'")
