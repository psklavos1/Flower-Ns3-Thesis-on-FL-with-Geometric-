# third party
import tensorflow as tf
from tensorflow import keras
import flwr as fl
# local
from protos import metric_service_pb2

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def on_train_batch_end(self, batch, logs=None):
        l2_norm = self.model.train_l2_norm.result()
        # Set up a gRPC client
        stub = self.model.stub
        # Send the metric to the server and get the response
        response = stub.SendMetricAndWait(
            metric_service_pb2.MetricRequest(l2_norm=l2_norm)
        )
        # Determine if the round has been stopped
        if response.stop_round:
            print("Round Stopped")
            self.model.stop_training = True
