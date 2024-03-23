# third party
# import math
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Metrics,
)
from omegaconf import DictConfig

# built-in
from typing import (
    List,
    Tuple,
    Union,
    Dict,
)

# local
from utils.monitor import Monitor
from server.custom_client_manger import CustomClientManager
from server.grpc_metric_client import GRPCMetricClient
from network.network import Network
from network.ns3_round import Ns3_Round


class FedAvgWithGeometric(FedAvg):
    """
    @extends flwr.server.strategy.FedAvg
    A custom strategy implementation to manage all the server side actions in each step of the federated learning.
    In Flower the strategies consist the main tool to modify the functionality and the logic of server side management.
    This custom implementation used a modified FedAvg logic to include FDA in the functionality. Not all aspects will
    be preseneted here as further information can be found in the documentation of FedAvg. The methods following are the,
    ones that where modified in this implementation.

    Methods:
        configure_fit(server_round, parameters, client_manager):
        aggregate_fit(server_round, results, failures):
    """

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
        # class variables
        self.network = ns3_network
        self.monitor = monitor
        self.metric_client = metric_client

        # Given From Ns3
        self.ns3_res = {}
        self.dropouts = []

    """
    Class Destructor to clean up grpc channel.
    """

    def __del__(self):
        # Clean up the gRPC client
        if self.metric_client is not None:
            self.metric_client.close()

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: CustomClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:

        """
        @override
        Configures the next round of training.
        Additional logic applied: The ns3_round simulation is run and in case of a client failing to communicate due to network related issued,
        the client is dropped and is not participating in the next training round.
        For more information refer to the flwr.server.strategy documentation.

        Parameters:
            server_round (int): The server round.
            parameters(Parameters): model parameters.
            client_manager (CustomClientManager): The client manager used in the application.
        Returns:
            List[Tuple[ClientProxy, FitIns]]: a list containing tuples of client proxies and their respective instructions.
        """
        fit_clients = super().configure_fit(server_round, parameters, client_manager)
        # Ns3 results
        self.ns3_res = self._ns3_simulation(fit_clients, server_round)

        #  If client is a dropout. Dont participate in training
        self.dropouts.clear()
        for i, (client_proxy, _) in enumerate(fit_clients):
            if self.ns3_res[self.monitor.get_index(client_proxy.cid)]["dropout"] == 1:
                self.dropouts.append(client_proxy)
                fit_clients.pop(i)

        num_fit_clients = len(fit_clients)
        self.monitor.set_fit_clients(num_fit_clients)
        # self.monitor.set_eval_clients(0)

        self.monitor.set_round(server_round)
        return fit_clients

    """
    @override
    Aggregates fit results using weighted average.
    Additional logic for passing in results the ns3 calculated statistics.
    
    Parameters:
        server_round (int): The server round.
        results (List[Tuple[ClientProxy, FitRes]]): a list containing the tuples of the client proxies along with their results.
        failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]: a list containing the tuples of the client proxies along with their results, accompanied by the occured Exception type.
    Returns:
        List[Union[Tuple[ClientProxy, FitRes], BaseException]]: a list containing either the resilts or failures depending on the execution's return.
    """

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

        # empty_metrics = {
        #     "done_processing": False,
        #     "l2_norm": 0,
        #     "roundTime": 0,
        #     "throughput": 0,
        #     "dropout": 1,
        # }
        # empty_parameters = Parameters(tensors=[], tensor_type="empty parameter")
        # for i in range(len(self.dropouts)):
        #     failures.append((DropoutException))

        return super().aggregate_fit(server_round, results, failures)

    """
    @override
    Utility function that prepares and call for an ns3 round simulation.
    
    Parameters:
        fit_clients (List[Tuple[ClientProxy, FitIns]]): The clients used in training.
        server_round (int): the round.
        
    Returns:
        Dict[int, Dict[str, float]]: A dictionary mapping the client ids to the corresponding dictionary of round
            resutls of each client.
    """

    def _ns3_simulation(
        self, fit_clients: List[Tuple[ClientProxy, FitIns]], server_round: int
    ) -> Dict[int, Dict[str, float]]:

        num_fit_clients = len(fit_clients)
        # Array of clients participating in training in format [0,4].
        clients = [
            self.monitor.get_index(fit_clients[i][0].cid)
            for i in range(num_fit_clients)
        ]

        # ns3 Simulation
        print("===================== NS3 Round Simulation =====================")
        ns3_round = Ns3_Round(self.network, clients, server_round)
        ns3_res = ns3_round.round_exec()
        print("================================================================")
<<<<<<< Updated upstream

        max_round_time = max(
            entry["downlinkTime"] + entry["uplinkTime"] + entry["computationTime"]
            for entry in ns3_res.values()
        )
        self.t_end = self.t_start + max_round_time

        # TODO Fix Time
        ns3_round.update_aggregate_time()
        # Preparation for next round
        self.t_start = self.t_end
=======
>>>>>>> Stashed changes
        return ns3_res


# ===================================================================================
# * Custom Exception


class DropoutException(Exception):
    """Exception raised for client dropouts in federated learning."""

    def __init__(self, message="Client dropped out"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"DropoutException: {self.message}"


# ===================================================================================
# * Utility Functions to configure the strategy


def get_fit_config_fn(config: DictConfig):
    """
    Generates the function to get fit_config.

    Parameters:
        config(DictConfig): Configuration file.

    Returns: The function to be use for fit_config."""

    def fit_config_fn(server_round: int):
        ## I could pass the round or change values in config depending on the round
        return {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "verbose": config.verbose,
        }

    return fit_config_fn


def get_eval_config_fn(config: DictConfig):
    """
    Generates the function to get eval_config.

    Parameters:
        config(DictConfig): Configuration file.

    Returns: The function to be use for eval_config."""

    def eval_config_fn(server_round: int):
        ## I could pass the round or change values in config depending on the round
        return {
            "batch_size": config.batch_size,
            "verbose": config.verbose,
        }

    return eval_config_fn


<<<<<<< Updated upstream
def metric_handlig(data):
    # data: Tuple (int: num_samples, dict: results from aggregate_fit)
=======
def metric_handlig(data: List[Tuple[int, Dict[str, Metrics]]]) -> Dict[str, Metrics]:
    """
    The rule on how to aggregate the results coming from the training of all clients in each round.

    Parameters:
        data (List[Tuple[int, FitRes]]): The tuple of each data entry contains first the number of samples and second a dict with all the metrics.

    Returns:
        Dict[str, Metrics]: the aggregated metrics.
    """
>>>>>>> Stashed changes
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
    computation_time_sum = sum(res_dict["computationTime"] for _, res_dict in data)
    avg_computation_time = (
        float(computation_time_sum / count_data) if count_data > 0 else float("nan")
    )

    # round time
    round_time = max(
        res_dict["downlinkTime"] + res_dict["computationTime"] + res_dict["uplinkTime"]
        for _, res_dict in data
    )

    # throughput
    total_throughput = sum(res_dict["throughput"] for _, res_dict in data)
    avg_throughput = (
        float(total_throughput / count_data) if count_data > 0 else float("nan")
    )

    return {
        "average_norm": avg_l2_norm,
        "round_time": round_time,
        "average_downlink_time": avg_downlink_time,
        "average_computation_time": avg_computation_time,
        "average_uplink_time": avg_uplink_time,
        "average_throughput": avg_throughput,
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
