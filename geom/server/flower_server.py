# third party
import time
import flwr as fl

# from flwr.server.strategy import FedAvg

# local
from server.strategy import (
    FedAvgWithGeometric,
    get_eval_config_fn,
    get_fit_config_fn,
    metric_handlig,
    weighted_average,
)
from server.custom_client_manger import CustomClientManager
from server.grpc_metric_client import GRPCMetricClient
from network.network import Network

""" Flower Server logic """
class FlowerServer:
    def __init__(self, cfg, ns3_cfg, monitor):
        self.cfg = cfg
        self.ns3_cfg = ns3_cfg
        self.monitor = monitor

    def start(self):
        self.client_manager = CustomClientManager(self.monitor)
        ns3_network = Network(self.ns3_cfg)
        visualize = self.ns3_cfg.visualize
        ns3_network.start_ns3(visualize=visualize)
        # Wait fot Simulator to start operating
        time.sleep(self.ns3_cfg.sleep)
        ns3_network.connect()

        self.strategy = FedAvgWithGeometric(
            GRPCMetricClient(self.cfg.metric_server_address),
            monitor=self.monitor,
            ns3_network=ns3_network,
            cfg_clients=self.ns3_cfg.clients,
            fit_metrics_aggregation_fn=metric_handlig,
            evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
            on_fit_config_fn=get_fit_config_fn(self.cfg.fit_cfg),
            on_evaluate_config_fn=get_eval_config_fn(self.cfg.eval_cfg),
        )

        fl.server.start_server(
            server_address=self.cfg.flower_server_address,
            config=fl.server.ServerConfig(num_rounds=self.cfg.num_rounds),
            strategy=self.strategy,
            client_manager=self.client_manager,
        )
        ns3_network.disconnect()
