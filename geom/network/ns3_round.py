# built-in
from typing import (
    Dict,
    List,
)

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
        Call for an Ns3 round execution and returns the resutls utilizing the socket
        connection between ns3 and flower.

        Returns:
            A dictionary mapping the client ids to the corresponding dictionary of round
            resutls of each client.
        """
        ret_dict = {}

        parsed_clients = self.network.parse_clients(self.clients)
        net_sim_data = self.network.sendRequest(requestType=1, array=parsed_clients)

        ret_dict = net_sim_data
        for client in self.clients:
            # If dropped. FAILURE
            if (
                net_sim_data[client]["downlinkTime"] <= 0
                or net_sim_data[client]["uplinkTime"] <= 0
            ):
                ret_dict[client]["dropout"] = 1
                self.dropouts.append(1)
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

        if len(self.throughputs) > 0:
            self.avg_throughput = sum([t for t in self.throughputs]) / len(
                self.throughputs
            )

        print("=================== Ns3 Stats ===================")
        self.total_latency = max(self.latencies)
        print(f"Total Time in Latencies: {self.total_latency}")
        print(f"Average throughput: {self.avg_throughput}")
        print(f"Dropouts: {self.dropouts}")
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
