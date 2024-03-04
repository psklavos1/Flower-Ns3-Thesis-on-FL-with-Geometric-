# third party
import random
import tensorflow as tf
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
        self.epoch_pointer = 0

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
                    dataset_name, bias_template, partition_id
                )

            self.local_trainset, self.local_testset = get_dataset(
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

        x_train, y_train = self.local_trainset
        metrics = {}

        # Round start resets
        self.model.train_l2_norm.set_weight_mean(self.model.trainable_weights)
        self.model.resume_train()
        
        # Train the local model
        processed_samples, self.batch_pointer,self.epoch_pointer, l2_norm = self.model.fit(
            x_train,
            y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            batch_pointer=self.batch_pointer,
            epoch_pointer = self.epoch_pointer,
            verbose=verbose,
            callbacks=[CustomCallback(self.model)],
        )

        metrics["l2_norm"] = l2_norm
        metrics["done_processing"] = processed_samples == len(x_train)

        return self.model.get_weights(), processed_samples, metrics

    def evaluate(self, parameters, config):
        print("Evaluate")
        # self.model.summary()
        # self.model.get_pickle_size()
        # Decrypt config:
        verbose, batch_size = config["verbose"], config["batch_size"]
        x_test, y_test = self.local_testset
        # Set the weights to the ones sent by the global server
        self.model.set_weights(parameters)
        res = self.model.evaluate(
            x_test, y_test, batch_size=batch_size, verbose=verbose
        )
        _, _, loss, accuracy = res
        metrics = {"accuracy": accuracy}
        return loss, len(x_test), metrics


# * Networking
# def set_link(self, config):

#     # Set the Gaussian distribution for link speed in Kbytes
#     self.speed_min = config.link.min
#     self.speed_max = config.link.max
#     self.speed_mean = random.uniform(self.speed_min, self.speed_max)
#     self.speed_std = config.link.std
#     self.model_size = self.model.get_size()
#     # Set estimated delay
#     self.est_latency = self.model_size / self.speed_mean

# def set_delay(self):
#     # Set the link speed and delay for the upcoming run
#     link_speed = random.normalvariate(self.speed_mean, self.speed_std)
#     link_speed = max(min(link_speed, self.speed_max), self.speed_min)
#     self.delay = self.model_size / link_speed  # upload delay in sec
