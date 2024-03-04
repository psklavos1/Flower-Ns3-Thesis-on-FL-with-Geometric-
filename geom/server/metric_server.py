# third party
import grpc

# built-in
import threading
from concurrent import futures

# local
from protos import metric_service_pb2
from protos import metric_service_pb2_grpc
from utils.monitor import Monitor


class MetricServer(metric_service_pb2_grpc.MetricServiceServicer):
    def __init__(self, cfg, monitor: Monitor):
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
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.thread_max_workers)
        )
        metric_service_pb2_grpc.add_MetricServiceServicer_to_server(self, server)
        server.add_insecure_port(self.cfg.metric_server_address)
        server.start()
        server.wait_for_termination()

    def SendMetricAndWait(self, request, context):
        num_clients = max(
            self.monitor.get_eval_clients(), self.monitor.get_fit_clients()
        )
        with self.metrics_lock:
            self.client_metrics.append(request.l2_norm)
            if len(self.client_metrics) != num_clients:
                self.condition.wait()
            else:
                self.condition.notify_all()
            stop_round = any(norm > self.threshold for norm in self.client_metrics)
            self.processed_clients += 1
            if self.processed_clients == num_clients:
                self.batch_reset()
        return metric_service_pb2.MetricResponse(stop_round=stop_round)

    def batch_reset(self):
        self.client_metrics.clear()
        self.processed_clients = 0

    def update_threshold(self, threshold):
        self.threshold = threshold

    def ResetRoundState(self, request, context):
        self.client_metrics.clear()
        self.round = request.round
        self.update_threshold(self.threshold * self.threshold_discount_factor)
        return metric_service_pb2.Empty()
