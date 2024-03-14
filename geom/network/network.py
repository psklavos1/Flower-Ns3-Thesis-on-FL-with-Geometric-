# from py_interface import *
from ctypes import *
import socket
import struct
import subprocess
import os


class Network(object):
    def __init__(self, config):
        self.tcp_ip = config.tcp_ip
        self.port = config.port
        self.path = config.path
        self.program = config.program
        self.num_clients = config.clients.total
        self.clients_for_fit = config.clients.for_fit
        self.network_type = config.network_type
        self.server_type = config.server
        self.data_rate = config.data_rate
        self.model_type = config.model_type
        self.device_type = config.device_type
        self.net_cfg = (
            config.network.wifi
            if self.network_type == "wifi"
            else self.config.network.ethernet
        )

        if config.model_type == "mnist":
            self.model_size = config.model.mnist.size
        elif config.model_type == "fashion_mnist":
            self.model_size = config.model.fashion_mnist.size
        elif config.model_type == "cifar10":
            self.model_size = config.model.cifar10.size
        elif config.model_type == "cifar100":
            self.model_size = config.model.cifar100.size
        else:
            raise ValueError(f"Error Server Config for model type {config.model_type}")

        self.delays = []
        self.throughputs = []
        self.dropouts = 0

    def start_ns3(self, visualize=False):
        # Assuming the script is in the project's home directory
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        # Change the working directory
        os.chdir(cur_dir)

        # Confirm the current working directory
        print(f"\nCurrent Working Directory: {cur_dir}")
        print("==================== NS3 Network =================")

        if not self.is_configured():
            print("Project configure")
            proc = subprocess.Popen(
                "./waf configure",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=self.path,
            )
            # Read and print output and errors
            while True:
                output = proc.stdout.readline()
                if output == "" and proc.poll() is not None:
                    break
                if output:
                    print(output.strip())

            rc = proc.poll()
            print(f"Process exited with code {rc}")

            proc.wait()

            if proc.returncode != 0:
                print("A problem occured while configuring project")
                exit(-1)

            print("Configure complete!\n\n\n")

        print("Wait For Buld to Complete. Might take a while\n")
        proc = subprocess.Popen(
            "./waf build",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=self.path,
        )
        # Read and print output and errors
        while True:
            output = proc.stdout.readline()
            if output == "" and proc.poll() is not None:
                break
            if output:
                print(output.strip())

        rc = proc.poll()
        print(f"Process exited with code {rc}")

        proc.wait()

        if proc.returncode != 0:
            print("A problem occured while building")
            exit(-1)

        print("Build complete!\n\n\n", flush=True)

        command = (
            './waf -v --run "'
            + self.program
            + " --numClients="
            + str(self.num_clients)
            + " --networkType="
            + self.network_type
        )

        command += " --modelSize=" + str(self.model_size)

        """print(self.config.network)
        for net in self.config.network:s
            if net == self.network_type:
                print(net.items())"""

        if self.network_type == "wifi":
            command += " --txGain=" + str(self.net_cfg.tx_gain)
            command += " --maxPacketSize=" + str(self.net_cfg.max_packet_size)
        else:  # else assume ethernet
            command += " --maxPacketSize=" + str(self.net_cfg.max_packet_size)

        command += " --learningModel=" + str(self.server_type)
        command += " --dataRate=" + str(self.data_rate)
        command += " --deviceType=" + str(self.device_type)
        command += " --modelType=" + str(self.model_type)

        command += '" --vis' if visualize else '"'
        title = "NS3"
        print(f"Command to execute: {command}")
        terminal_command = f"gnome-terminal --title={title} -- {command}"

        proc = subprocess.Popen(
            terminal_command,
            shell=True,
            cwd=self.path,
        )

    def is_configured(self) -> bool:
        # Path to a file or directory that indicates configuration is done
        config_marker = os.path.join(self.path, "build/config.log")

        # Only run configuration if the marker doesn't exist
        if not os.path.exists(config_marker):
            return False
        return True

    # * Getters & Setteres
    # Getters
    def get_delays(self):
        return self.delays

    def get_througputs(self):
        return self.throughputs

    def get_dropouts(self):
        return self.dropouts

    # Setters
    def set_delays(self, x):
        self.delays = x

    def set_throughputs(self, x):
        self.throughputs = x

    def set_dropouts(self, x):
        self.dropouts = x

    def parse_clients(self, clients):
        clients_to_send = [0 for _ in range(self.num_clients)]
        for index in clients:
            clients_to_send[index] = 1
        return clients_to_send

    def connect(self):
        print(f"Ns3_connect(): Tcp Ip: {self.tcp_ip}, Port: {self.port}")
        self.socket = socket.create_connection(
            (
                self.tcp_ip,
                self.port,
            )
        )

    def sendRequest(self, *, requestType: int, array: list):
        print("Sending fit clients list.")
        print(array)
        print("Waiting for response")
        message = struct.pack("II", requestType, len(array))
        self.socket.send(message)
        # for the total number of clients
        # is the index in list at client.id equal
        for ele in array:
            self.socket.send(struct.pack("I", ele))

        # Verify received
        resp = self.socket.recv(8)

        if len(resp) < 8:
            print(len(resp), resp)

        command, nItems = struct.unpack("II", resp)
        ret = {}
        for i in range(nItems):
            dr = self.socket.recv(8 * 4)
            (
                eid,
                throughput,
                downlinkTime,
                uplinkTime,
            ) = struct.unpack("Qddd", dr)

            temp = {
                "downlinkTime": downlinkTime,
                "uplinkTime": uplinkTime,
                "throughput": throughput,
            }
            ret[eid] = temp
        print("Response received")

        empty = "        "
        for key, val in ret.items():
            print(f"id {key}:")  # Print the id followed by a colon
            for vkey, vval in val.items():
                # Indent and print each key-value pair
                formatted_key = vkey.ljust(10)
                print(f"{empty}{formatted_key} -> {vval:.2f}")

        return ret

    def sendAsyncRequest(self, *, requestType: int, array: list):
        print("sending")
        print(array)
        message = struct.pack("II", requestType, len(array))
        self.socket.send(message)
        # for the total number of clients
        # is the index in lit at client.id equal
        for ele in array:
            self.socket.send(struct.pack("I", ele))

    def readAsyncResponse(self):
        resp = self.socket.recv(8)
        print("resp")
        print(resp)
        if len(resp) < 8:
            print(len(resp), resp)
        command, nItems = struct.unpack("II", resp)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(command)
        if command == 3:
            return "end"
        ret = {}
        for i in range(nItems):
            dr = self.socket.recv(8 * 4)
            eid, startTime, endTime, throughput = struct.unpack("Qddd", dr)
            temp = {
                "startTime": startTime,
                "endTime": endTime,
                "throughput": throughput,
            }
            ret[eid] = temp
        return ret

    def disconnect(self):
        # self.socketendAsyncRequest(requestType=2, array=[])
        self.socket.close()
