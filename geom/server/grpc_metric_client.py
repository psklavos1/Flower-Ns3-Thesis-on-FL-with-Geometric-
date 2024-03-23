# third party
import grpc

# local
from protos import metric_service_pb2
from protos import metric_service_pb2_grpc


class GRPCMetricClient:
    """
    The gRPC Metric Client can call methods on the server as if they were local method calls.
    Clients need to know the address of the server and the service methods they want to call.

    Methods:
        reset_round_state(round): Resets the round tracking state at the start of a round.
        close(): Closes the connection to the server.
    """

    def __init__(
        self,
        address,
    ):
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = metric_service_pb2_grpc.MetricServiceStub(self.channel)

    def reset_round_state(self, round: int):
        """
        To be used at the start of a training round to reset the tracking logic.

        Parameters:
            round: the round number used if needed for logic in regards with the threshold monitoring.
        """
        self.stub.ResetRoundState(metric_service_pb2.RoundRequest(round=round))

    def close(self):
        """
        To be used for closing the channel providing the grpc connection.
        """
        self.channel.close()
