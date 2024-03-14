from network.network import Network
from typing import List


class Ns3_Round:
    def __init__(self, network: Network, clients: List[int], round: int) -> None:
        # Statistics derived from ns3
        self.round = round
        self.network = network
        self.throughputs = []
        self.latencies = []
        self.dropouts = []
        self.clients = clients
        self.processed_clients = []
        self.total_round_time = 0.0
        self.avg_throughput = 0.0
        self.start_time = 0.0
        self.aggregate_time = 0.0

    def round_exec(self, t_start) -> float:

        # {0: {'roundTime', 'throughput', 'dropout'}, ...}
        ret_dict = {}
        self.start_time = t_start
        # parse: init format[0,2,3] -> parsed format[1,0,1,1]
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

        print("================ Ns3 Stats ================")
        self.total_latency = max(self.latencies)
        print(f"Total Time in Latencies: {self.total_latency}")
        print(f"Average throughput: {self.avg_throughput}")
        print(f"Dropouts: {self.dropouts}")
        print("===========================================\n")

        return ret_dict

    # Getters and setters
    # Getters
    def get_throughputs(self):
        return self.throughputs

    def get_latencies(self):
        return self.latencies

    def get_total_round_time(self):
        return self.total_round_time

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

    def get_start_time(self):
        return self.start_time

    def get_aggregate_time(self):
        return self.aggregate_time

    # Setters
    def set_start_time(self, x):
        self.start_time = x

    def update_aggregate_time(self):
        """A round must have been executed first"""
        self.aggregate_time = self.total_round_time + self.start_time
