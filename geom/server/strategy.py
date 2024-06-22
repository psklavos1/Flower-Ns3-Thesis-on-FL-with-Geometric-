# third party
# import math
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitIns,
    FitRes,
    EvaluateRes,
    Parameters,
    Metrics,
    Scalar,
)
from omegaconf import DictConfig

# built-in
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Optional,
)
import json

# local
from utils.monitor import Monitor
from server.custom_client_manger import CustomClientManager
from server.grpc_metric_client import GRPCMetricClient
from network.network import Network
from network.ns3_round import Ns3_Round
from utils.model import CustomModel
from dataframe.data_manager import DataManager


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
        cfg,
        root_dir="csv_logs",
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            fraction_fit=0.0001,
            fraction_evaluate=0.0001,
            min_fit_clients=cfg.clients.for_fit,
            min_evaluate_clients=cfg.clients.for_eval,
            min_available_clients=cfg.clients.total,
        )
        # class variables
        self.network = ns3_network
        self.monitor = monitor
        self.metric_client = metric_client
        self.data_manager = DataManager(log_dir=str(root_dir) + "/csv_logs")

        info = {
            "algorithm": "synchronous" if cfg.threshold == 0 else "fda",
            "num_clients": cfg.num_clients,
            "dataset": cfg.dataset,
            "ann": cfg.ann,
            "threshold": cfg.threshold,
            "thres_discount_factor": cfg.thres_discount_factor,
            "steps_threshold": cfg.steps_threshold,
            "train_batch_size": cfg.fit_cfg.batch_size,
            "eval_batch_size": cfg.eval_cfg.batch_size,
            "network_template": cfg.wifi_net_template,
            "client_mobility": cfg.moving_clients,
            "non_iid": cfg.non_iid,
        }
        if cfg.non_iid:
            info["bias_template"] = cfg.bias_template
        self.data_manager.add_info_data(info)
        self.data_manager.save_info_data()
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
        fit_clients_copy = fit_clients.copy()

        for i, (client_proxy, _) in enumerate(fit_clients_copy):
            if self.ns3_res[self.monitor.get_index(client_proxy.cid)]["dropout"] == 1:
                self.dropouts.append(client_proxy)
                fit_clients.remove(
                    (client_proxy, _)
                )  # Use remove() to handle element removal

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
        # * Append to Results dropouts
        for client, res in results:
            for id, data_dict in self.ns3_res.items():
                if self.monitor.get_cid(id) == client.cid:
                    res.metrics.update(data_dict)
                    # ? Not beutiful way to handle dropout, but fastest solution to keep track
                    res.metrics["dropout"] = len(self.dropouts)

        agg_fit_res = super().aggregate_fit(server_round, results, failures)

        # * Take care of server dataframe fit round results
        _, metrics_aggregated = agg_fit_res
        srvr_data = {"round": server_round}
        srvr_data.update(metrics_aggregated)
        self.data_manager.add_server_data(srvr_data)

        return agg_fit_res

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        loss, metrics = super().evaluate(server_round, parameters)

        # to initialize client
        if server_round == 0:
            self.data_manager.add_server_data(
                {
                    "round": server_round,
                    "experiment_id": "exp_000000",
                    "centralized_loss": 0.0,
                    "centralized_acc": 0.0,
                    "distributed_loss": 0.0,
                    "distributed_acc": 0.0,
                    "dropouts": 0,
                    "average_norm": 0.0,
                    "round_time": 0.0,
                    "average_downlink_time": 0.0,
                    "average_rtc_check_time": 0.0,
                    "average_uplink_time": 0.0,
                    "average_computation_time": 0.0,
                    "average_communication_time": 0.0,
                    "average_fit_time": 0.0,
                    "average_throughput": 0.0,
                }
            )

        self.data_manager.append_server_data_to_last_row(
            {
                "centralized_loss": loss,
                "centralized_acc": metrics["accuracy"],
            }
        )

        return loss, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[float, Dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # At theis point all the data of the client training will be logged if this clients logs his res
        for client_proxy, eval_res in results:
            epoch_data = json.loads(eval_res.metrics["epoch_data"])
            batch_data = json.loads(eval_res.metrics["batch_data"])

            if epoch_data and batch_data:  # client logging results
                client_id = str(self.monitor.get_index(client_proxy.cid))
                if server_round == 1:
                    self._init_client_metric_ordering(client_id)

                self.data_manager.add_epoch_data(epoch_data, client_id)
                self.data_manager.save_epoch_data(client_id=client_id)

                self.data_manager.add_batch_data_list(batch_data, client_id)
                self.data_manager.save_batch_data(client_id=client_id)

        # Append to the fit res of the server the required eval res

        self.data_manager.append_server_data_to_last_row(
            {
                "distributed_loss": aggregated_loss,
                "distributed_acc": aggregated_metrics["accuracy"],
            }
        )
        # At this point everything has been processed so we store it
        self.data_manager.save_server_data()

        return aggregated_loss, aggregated_metrics

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
        return ns3_res

    def _init_client_metric_ordering(self, client_id: str):
        # Initialize ordering
        epoch_metrics_template = {
            "round": 0,  # int
            "experiment_id": "exp_000000",  # str
            "client_id": 0,  # int
            "epoch_steps": 0,  # int
            "total_steps": 0,  # int
            "l2_norm": 0.0,  # float
            "total_computation_time": 0.0,  # float
            "total_rtc_check_time": 0.0,  # float
            "total_duration": 0.0,  # float
            "val_loss": 0.0,  # float
            "val_accuracy": 0.0,  # float
        }
        self.data_manager.add_epoch_data(epoch_metrics_template, client_id)

        step_metrics_template = {
            "round": 0,  # int
            "experiment_id": "exp_000000",  # str
            "client_id": 0,  # int
            "step_no": 0,  # int
            "loss": 0.0,  # float
            "accuracy": 0.0,  # float
            "l2_norm": 0.0,  # float
            "computation_time": 0.0,  # float
            "rtc_check_time": 0.0,  # float
            "batch_time": 0.0,  # float
        }
        self.data_manager.add_batch_data(step_metrics_template, client_id)


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
            "round": server_round,
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
            "round": server_round,
        }

    return eval_config_fn


def get_evaluate_fn(testset, dataset, ann):

    def evaluate_fn(server_round: int, parameters, config):
        model = CustomModel(ds_name=dataset, ann_name=ann)
        model.set_weights(parameters)
        res = model.evaluate(testset, batch_size=256, verbose=1)
        return res["val_loss"], {"accuracy": res["val_accuracy"]}

    return evaluate_fn


def metric_handlig(data: List[Tuple[int, Dict[str, Metrics]]]) -> Dict[str, Metrics]:
    """
    The rule on how to aggregate the results coming from the training of all clients in each round.

    Parameters:
        data (List[Tuple[int, FitRes]]): The tuple of each data entry contains first the number of samples and second a dict with all the metrics.

    Returns:
        Dict[str, Metrics]: the aggregated metrics.
    """
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

    # avg time in monitoring RTC
    rtc_check_time_sum = sum(res_dict["rtc_check_time"] for _, res_dict in data)
    avg_rtc_check_time = (
        float(rtc_check_time_sum / count_data) if count_data > 0 else float("nan")
    )

    # avg computation time
    computation_time_sum = sum(res_dict["computation_time"] for _, res_dict in data)
    avg_computation_time = (
        float(computation_time_sum / count_data) if count_data > 0 else float("nan")
    )

    # avg communication time
    communication_time_sum = uplink_time_sum + downlink_time_sum + rtc_check_time_sum
    avg_communication_time = (
        float(communication_time_sum / count_data) if count_data > 0 else float("nan")
    )

    # avg fit time
    fit_time_sum = sum(res_dict["fit_duration"] for _, res_dict in data)
    avg_fit_time = float(fit_time_sum / count_data) if count_data > 0 else float("nan")

    # round time
    round_time = max(
        res_dict["downlinkTime"] + res_dict["fit_duration"] + res_dict["uplinkTime"]
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
        "average_rtc_check_time": avg_rtc_check_time,
        "average_uplink_time": avg_uplink_time,
        "average_computation_time": avg_computation_time,
        "average_communication_time": avg_communication_time,
        "average_fit_time": avg_fit_time,
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
