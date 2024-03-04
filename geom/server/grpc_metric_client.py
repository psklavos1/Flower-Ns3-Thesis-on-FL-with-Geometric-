# third party
import grpc

# local
from protos import metric_service_pb2
from protos import metric_service_pb2_grpc


class GRPCMetricClient:
    def __init__(
        self,
        address,
    ):
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = metric_service_pb2_grpc.MetricServiceStub(self.channel)

    def reset_round_state(self, round):
        self.stub.ResetRoundState(metric_service_pb2.RoundRequest(round=round))

    def close(self):
        self.channel.close()
