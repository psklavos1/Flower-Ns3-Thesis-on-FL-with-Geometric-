# third party
import math
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    Parameters,
    Metrics,
)
from omegaconf import DictConfig

# built-in
from typing import (
    List,
    Tuple,
    Union,
)

# local
from utils.monitor import Monitor
from server.custom_client_manger import CustomClientManager
from server.grpc_metric_client import GRPCMetricClient
from network.network import Network
from network.ns3_round import Ns3_Round


class FedAvgWithGeometric(FedAvg):
    """Custom FedAvg"""

    def __init__(
        self,
        metric_client: GRPCMetricClient,
        monitor: Monitor,
        ns3_network: Network,
        cfg_clients,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            fraction_fit=0.0001,
            fraction_evaluate=0.0001,
            min_fit_clients=cfg_clients.for_fit,
            min_evaluate_clients=cfg_clients.for_eval,
            min_available_clients=cfg_clients.total,
        )
        # TODO careful track time
        self.t_start = 0.0
        self.t_end = 0.0
        self.network = ns3_network
        self.monitor = monitor
        self.metric_client = metric_client

        # * Given From Ns3
        self.ns3_res = {}
        self.dropouts = []

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: CustomClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:

        fit_clients = super().configure_fit(server_round, parameters, client_manager)
        # Ns3 results
        self.ns3_res = self.ns3_simulation(fit_clients, server_round)
        # * If client is a dropout. Dont participate in training
        self.dropouts.clear()
        for i, (client_proxy, _) in enumerate(fit_clients):
            if self.ns3_res[self.monitor.get_index(client_proxy.cid)]["dropout"] == 1:
                self.dropouts.append(client_proxy)
                fit_clients.pop(i)

        num_fit_clients = len(fit_clients)
        self.monitor.set_fit_clients(num_fit_clients)
        self.monitor.set_round(server_round)
        return fit_clients

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        ret = super().configure_evaluate(server_round, parameters, client_manager)
        self.monitor.set_eval_clients(len(ret))
        self.monitor.set_round(server_round)
        return ret

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        self.metric_client.reset_round_state(server_round)
        self.monitor.set_fit_clients(0)
        # results[id_client][ClientProxy==0, FitRes== 1],
        # * Append to Results
        for client, res in results:
            for id, data_dict in self.ns3_res.items():
                if self.monitor.get_cid(id) == client.cid:
                    res.metrics.update(data_dict)
                    # ? Not beutiful way to handle dropout, but fastest solution to keep track
                    res.metrics["dropout"] = len(self.dropouts)

        return super().aggregate_fit(server_round, results, failures)

    def __del__(self):
        # Clean up the gRPC client
        if self.metric_client is not None:
            self.metric_client.close()

    def ns3_simulation(self, fit_clients, server_round):
        num_fit_clients = len(fit_clients)
        # print(num_fit_clients)
        # input()

        # * clients: is an array of clients that participate in fit
        # * in the format [0, 2, 3] which is the representation of
        # * a cid to the format that is used in ns3 [1,0,1,1]
        clients = [
            self.monitor.get_index(fit_clients[i][0].cid)
            for i in range(num_fit_clients)
        ]

        # ns3 Simulation
        print("===================== NS3 Round Simulation =====================")
        ns3_round = Ns3_Round(self.network, clients, server_round)
        # The order here is with increasing index.
        ns3_res = ns3_round.round_exec(self.t_start)
        print("================================================================")

        return ns3_res


class DropoutException(Exception):
    """Exception raised for client dropouts in federated learning."""

    def __init__(self, message="Client dropped out"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"DropoutException: {self.message}"


# Out of Class
def get_fit_config_fn(config: DictConfig):
    def fit_config_fn(server_round: int):
        ## I could pass the round or change values in config depending on the round
        return {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "verbose": config.verbose,
        }

    return fit_config_fn


def get_eval_config_fn(config: DictConfig):
    def eval_config_fn(server_round: int):
        ## I could pass the round or change values in config depending on the round
        return {
            "batch_size": config.batch_size,
            "verbose": config.verbose,
        }

    return eval_config_fn


def metric_handlig(data):
    # data: Tuple (int: num_samples, dict: results from aggregate_fit)
    # (1600, {'done_processing': False, 'l2_norm': 25.036333084106445, 'roundTime': 10.78390491, 'throughput': 1179.1668024261412, 'dropout': 0})

    count_data = len(data)
    # l2_norm
    l2_norm_sum = sum(res_dict["l2_norm"] for _, res_dict in data)
    avg_l2_norm = float(l2_norm_sum / count_data) if count_data > 0 else float("nan")

    # avg downlink time
    downlink_time_sum = sum(res_dict["downlinkTime"] for _, res_dict in data)
    avg_downlink_time = (
        float(downlink_time_sum / count_data) if count_data > 0 else float("nan")
    )

    # avg uplink time
    uplink_time_sum = sum(res_dict["uplinkTime"] for _, res_dict in data)
    avg_uplink_time = (
        float(uplink_time_sum / count_data) if count_data > 0 else float("nan")
    )

    # avg computation time
    computation_time_sum = sum(res_dict["computation_time"] for _, res_dict in data)
    avg_computation_time = (
        float(computation_time_sum / count_data) if count_data > 0 else float("nan")
    )
    # round time
    # TODO: add time tracking componennt for training only
    round_time = max(
        res_dict["downlinkTime"] + res_dict["computation_time"] + res_dict["uplinkTime"]
        for _, res_dict in data
    )

    # throughput
    total_throughput = sum(res_dict["throughput"] for _, res_dict in data)
    avg_throughput = (
        float(total_throughput / count_data) if count_data > 0 else float("nan")
    )

    # Dropouts
    total_dropouts = data[0][1]["dropout"]

    return {
        "average_norm": avg_l2_norm,
        "round_time": round_time,
        "average_downlink_time": avg_downlink_time,
        "average_computation_time": avg_computation_time,
        "average_uplink_time": avg_uplink_time,
        "average_throughput": avg_throughput,
        "dropouts": total_dropouts,
    }


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate with weighted average during evaluation.

    Parameters:
    ----------
    metrics : List[Tuple[int, Metrics]]
        A list of tuples where each tuple contains the number of examples processed by a client and its reported metrics.

    Returns:
    -------
    Metrics
        The aggregated metrics.
    """

    total_examples = 0
    total_accuracy = 0.0

    for num_examples, client_metrics in metrics:
        client_accuracy = client_metrics["accuracy"]
        total_accuracy += client_accuracy * num_examples
        total_examples += num_examples

    # Calculate the weighted average accuracy
    aggregated_accuracy = total_accuracy / total_examples if total_examples > 0 else 0

    return {"accuracy": aggregated_accuracy}
