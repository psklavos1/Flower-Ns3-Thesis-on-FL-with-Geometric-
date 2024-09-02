# third party
import time
import flwr as fl
import tensorflow as tf
from flwr.common import Config
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# local
from .strategy import (
    FedAvgWithGeometric,
    get_eval_config_fn,
    get_fit_config_fn,
    metric_handlig,
    weighted_average,
    get_evaluate_fn,
)
from .custom_client_manger import CustomClientManager
from .grpc_metric_client import GRPCMetricClient
from network.network import Network
from utils.monitor import Monitor
from utils.dataset import get_testset


class FlowerServer:
    """
    A Server class used to manage the flower learning side, responsible for training and aggregation

    Methods:
        start(): Start the flower server and along with it initiate the experiment in flower and ns3 side.
    """

    def __init__(self, cfg: Config, ns3_cfg: Config, monitor: Monitor, root_dir):
        self.cfg = cfg
        self.ns3_cfg = ns3_cfg
        self.monitor = monitor
        self.dataset = cfg.dataset
        self.ann = cfg.ann
        self.root_dir = root_dir

        # Ensure TensorFlow uses the GPU if available
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        self.testset = get_testset(self.dataset)

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
            cfg=self.cfg,
            fit_metrics_aggregation_fn=metric_handlig,
            evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
            on_fit_config_fn=get_fit_config_fn(self.cfg.fit_cfg),
            on_evaluate_config_fn=get_eval_config_fn(self.cfg.eval_cfg),
            evaluate_fn=get_evaluate_fn(
                testset=self.testset,
                dataset=self.dataset,
                ann=self.ann,
            ),
            root_dir=self.root_dir,
        )

        hist = fl.server.start_server(
            server_address=self.cfg.flower_server_address,
            config=fl.server.ServerConfig(num_rounds=self.cfg.num_rounds),
            strategy=self.strategy,
            client_manager=self.client_manager,
            grpc_max_message_length=1024 * 1024 * 1024,  # <-- 1GB
        )
        print("Experiment Over")

        # print(f"This is history \n\n\n\n {hist}")
        # After server returns meaning the end of experiment. Close ns3 simulator connection.
        ns3_network.disconnect()
