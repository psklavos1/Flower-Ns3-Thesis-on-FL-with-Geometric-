# third party
import time
import flwr as fl
from flwr.common import Config


# local
from .strategy import (
    FedAvgWithGeometric,
    get_eval_config_fn,
    get_fit_config_fn,
    metric_handlig,
    weighted_average,
)
from .custom_client_manger import CustomClientManager
from .grpc_metric_client import GRPCMetricClient
from network.network import Network
from utils.monitor import Monitor


class FlowerServer:
    """
    A Server class used to manage the flower learning side, responsible for training and aggregation

    Methods:
        start(): Start the flower server and along with it initiate the experiment in flower and ns3 side.
    """

    def __init__(self, cfg: Config, ns3_cfg: Config, monitor: Monitor):
        self.cfg = cfg
        self.ns3_cfg = ns3_cfg
        self.monitor = monitor

    def start(self):
        """
        Start the flower server and along with it initiate the experiment in flower and ns3 side setting up the initial parameters
        """
        self.client_manager = CustomClientManager(self.monitor)
        ns3_network = Network(self.ns3_cfg, self.cfg.clients)

        visualize = self.ns3_cfg.visualize
        ns3_network.start_ns3(visualize=visualize)
        # Wait fot Simulator to start operating
        time.sleep(self.cfg.wait_ns3_establishment)
        ns3_network.connect()

        self.strategy = FedAvgWithGeometric(
            GRPCMetricClient(self.cfg.metric_server_address),
            monitor=self.monitor,
            ns3_network=ns3_network,
            cfg_clients=self.cfg.clients,
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
        # After server returns meaning the end of experiment. Close ns3 simulator connection.
        ns3_network.disconnect()
