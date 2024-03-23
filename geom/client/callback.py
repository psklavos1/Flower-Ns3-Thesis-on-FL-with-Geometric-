# third party
from tensorflow import keras

# local
from protos import metric_service_pb2
from utils.model import CustomModel


class CustomCallback(keras.callbacks.Callback):
    """
    @extends tensorflow.keras.callbacks.Callback
    Custom callbacks implemetations to use in client side during training

    Methods:
        on_train_batch_end(batch, logs): Called to initiate an action at the end of a training batch
    """

    def __init__(self, model: CustomModel) -> None:
        super().__init__()
        self.model = model

    def on_train_batch_end(self, batch: int, logs=None):
        """
        @override
        At the end of a training batch FDA logic is applied. In the following implementation each client
        sends the l2_norm of its local model to the server who decides whether to continue training or not.
        For more information about the parameters refer to the fl.client.NumPyClient documentation.

        Parameters:
            batch (int): the batch_no being processed
            logs (Any): passed to repclicate a state. Could be a dict of metrics or something else that neeeds processing.
        """

        l2_norm = self.model.train_l2_norm.result()

        # Send the metric to the server and get the response
        stub = self.model.stub
        response = stub.SendMetricAndWait(
            metric_service_pb2.MetricRequest(l2_norm=l2_norm)
        )
        # Determine if the round has been stopped
        if response.stop_round:
            print("Round Stopped")
            self.model.stop_training = True
