# third party
import flwr as fl

# built-in
import os

# local
from .callback import CustomCallback
from utils.dataset import get_dataset, generate_class_percentages
from .model import CustomModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
# print(tf.config.list_physical_devices("GPU"))

"""To federate create flower client"""


class FlexibleClient(fl.client.NumPyClient):
    # Init Just for this examples where we partition our own dataset, otherwise no need for num_clients and partition_id
    def __init__(
        self,
        num_clients=2,
        partition_id=0,
        ds_name="mnist",
        non_iid=False,
        bias_template=0,
    ):
        self.batch_pointer = 0

        # Map dataset names to their respective functions
        dataset_model_map = {
            "mnist": ("mnist", CustomModel),
            "fashion_mnist": ("fashion_mnist", CustomModel),
            "cifar10": ("cifar10", CustomModel),
            "cifar100": ("cifar100", CustomModel),
        }

        if ds_name in dataset_model_map:

            dataset_name, model_class = dataset_model_map[ds_name]

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

            self.model: CustomModel = model_class(ds_name=dataset_name)
        else:
            raise ValueError(f"Invalid dataset name '{ds_name}'")

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
        processed_samples = batches_in_epoch * self.batch_pointer
        metrics["l2_norm"] = l2_norm
        metrics["computation_time"] = computation_time
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
