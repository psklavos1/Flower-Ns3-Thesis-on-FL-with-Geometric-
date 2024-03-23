# third party
import grpc
from flwr.common import Config

# built-in
import threading
from concurrent import futures

# local
from protos import metric_service_pb2
from protos import metric_service_pb2_grpc
from utils.monitor import Monitor


class MetricServer(metric_service_pb2_grpc.MetricServiceServicer):
    """
    @extends metric_service_pb2_grpc.MetricServiceServicer
    A Server class used to manage the metric tracking logic, responsible for deciding when to stop a round of learning using FDA logic.
    This is the server side of a grpc connection, which means that implements the service interfaces defined in the protobuf files.
    The server listens for requests from clients on a specific port, and upon receiving a request, it executes the requested procedure
    and sends a response back to the client.

    Methods:
        start(): Start the grpc Metric Server, listening for requests after binding the given address.
        SendMetricAndWait(request, context): The grpc clients send their norms and await of a response on wheter to continue training or not based on the server's decision.
        ResetRoundState(request, context): Resets all the processed metrics to be ready for a new round.
    """

    def __init__(self, cfg: Config, monitor: Monitor):
        # Class variables
        self.monitor = monitor
        self.cfg = cfg
        self.threshold = cfg.threshold
        self.threshold_discount_factor = cfg.thres_discount_factor
        self.processed_clients = 0
        self.client_metrics = []
        self.metrics_lock = threading.Lock()
        self.thread_max_workers = cfg.thread_max_workers
        self.condition = threading.Condition(self.metrics_lock)

    def start(self):
        """
        Start the grpc Metric Server, listening for requests after binding the given address.
        """
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.thread_max_workers)
        )
        metric_service_pb2_grpc.add_MetricServiceServicer_to_server(self, server)
        server.add_insecure_port(self.cfg.metric_server_address)
        server.start()
        server.wait_for_termination()

    def SendMetricAndWait(self, request, context):
        """
        Parameters:
            request (float): the l2_norm of each client calling the gRPC function.

        Returns:
            MetricResponse-Boolean: True if to continue training
        """

        num_clients = self.monitor.get_fit_clients()
        # num_clients = max(
        #     self.monitor.get_eval_clients(), self.monitor.get_fit_clients()
        # )
        with self.metrics_lock:
            self.client_metrics.append(request.l2_norm)
            if len(self.client_metrics) != num_clients:
                self.condition.wait()
            else:
                self.condition.notify_all()
            stop_round = any(norm > self.threshold for norm in self.client_metrics)
            self.processed_clients += 1
            if self.processed_clients == num_clients:
                self._batch_reset()
        return metric_service_pb2.MetricResponse(stop_round=stop_round)

    def ResetRoundState(self, request, context):
        """
        Resets all the metric tracking logic to be ready to process a new round

        Parameters:
            request (int): the round

        Returns:
            Empty: Message Type defined in grpc context
        """
        self.client_metrics.clear()
        self.round = request.round
        self._update_threshold(self.threshold * self.threshold_discount_factor)
        return metric_service_pb2.Empty()

    def _batch_reset(self):
        """
        Resets the processed metrics to start processing a new batch.
        """
        self.client_metrics.clear()
        self.processed_clients = 0

    def _update_threshold(self, threshold):
        """
        Used to update the threshold to a new value
        Parameters:
            threshold (float): the new threshold value
        """
        self.threshold = threshold
