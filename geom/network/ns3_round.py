# built-in
from typing import (
    Dict,
    List,
)
import numpy as np

# local
from .network import Network


class Ns3_Round:
    """
    Keep track of an Ns3 Round.

    Methods:
        round_exec(): Calls for an Ns3 round execution and returns the resutls.
        getters: to retrieve various round statistics.
    """

    def __init__(self, network: Network, clients: List, round: int):
        # Class variables
        self.round = round
        self.network = network
        self.throughputs = []
        self.latencies = []
        self.dropouts = []
        self.clients = clients
        self.processed_clients = []
        self.avg_throughput = 0.0

    # ===============================================================================

    def round_exec(self) -> Dict[int, Dict[str, float]]:
        """
        Call for an Ns3 round execution and returns the results utilizing the socket
        connection between ns3 and flower.

        Returns:
            A dictionary mapping the client ids to the corresponding dictionary of round
            results of each client.
        """
        ret_dict = {}

        parsed_clients = self.network.parse_clients(self.clients)
        net_sim_data = self.network.sendRequest(requestType=1, array=parsed_clients)

        ret_dict = net_sim_data
        dropout_indices = []
        downlink_times = []

        for i, client in enumerate(self.clients):
            downlink_time = net_sim_data[client]["downlinkTime"]
            downlink_times.append(downlink_time)

            # If dropped. FAILURE
            if downlink_time <= 0 or net_sim_data[client]["uplinkTime"] <= 0:
                dropout_indices.append(i)
                ret_dict[client]["dropout"] = 1
                self.dropouts.append(1)
                self.throughputs.append(0)
                self.latencies.append(0)
                print(f"Client {str(client)} Dropped out due to network")
                continue

            # SUCCESS
            ret_dict[client]["dropout"] = 0
            self.dropouts.append(0)
            self.throughputs.append(net_sim_data[client]["throughput"])
            self.latencies.append(
                net_sim_data[client]["downlinkTime"]
                + net_sim_data[client]["uplinkTime"]
            )
            self.processed_clients.append(client)

        # Calculate the median of downlink times excluding initial dropouts
        valid_downlink_times = [
            downlink_times[i]
            for i in range(len(downlink_times))
            if i not in dropout_indices
        ]
        downlink_median = np.median(valid_downlink_times)

        # Define a fixed upper boundary for downlink time (e.g., 3 times the median)
        upper_boundary = downlink_median * 5

        # Identify outliers based on the upper boundary
        for i, client in enumerate(self.clients):
            downlink_time = net_sim_data[client]["downlinkTime"]
            if downlink_time > upper_boundary:
                print(f"Outlier: {downlink_time} > {upper_boundary}")

            # if downlink_time > upper_boundary:
            #     valid_mean = np.mean(valid_downlink_times)
            #     valid_std = np.std(valid_downlink_times)
            #     new_downlink_time = np.random.normal(loc=valid_mean, scale=valid_std)
            #     ret_dict[client]["downlinkTime"] = new_downlink_time

            #     # Update latency with the new downlink time
            #     self.latencies[i] = (
            #         new_downlink_time + net_sim_data[client]["uplinkTime"]
            #     )

            #     print(
            #         f"Client {str(client)} Downlink time adjusted to {new_downlink_time} due to high original value"
            #     )

        # Remove throughputs and latencies for the initial dropouts
        self.throughputs = [
            t for i, t in enumerate(self.throughputs) if i not in dropout_indices
        ]
        self.latencies = [
            l for i, l in enumerate(self.latencies) if i not in dropout_indices
        ]

        if len(self.throughputs) > 0:
            self.avg_throughput = sum(self.throughputs) / len(self.throughputs)

        print("=================== Ns3 Stats ===================")
        self.total_latency = max(self.latencies) if self.latencies else 0
        print(f"Total Time in Latencies: {self.total_latency}")
        print(f"Average throughput: {self.avg_throughput}")
        print(f"Dropouts: {self.dropouts}")
        print(f"Downlinks: {downlink_times}")
        print("=================================================\n")

        return ret_dict

    # ==========================================================================
    # * Getters
    def get_throughputs(self):
        return self.throughputs

    def get_latencies(self):
        return self.latencies

    def get_clients(self):
        return self.clients

    def get_processed_clients(self):
        return self.processed_clients

    def get_dropouts(self):
        return self.dropouts

    def get_avg_throughput(self):
        return self.avg_throughput

    def get_round(self):
        return self.round

    # ===========================================================================
