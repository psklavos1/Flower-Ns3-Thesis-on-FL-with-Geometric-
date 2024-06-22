# built-in
from ctypes import *
import socket
import struct
import subprocess
import os
from typing import List, Dict, Any


class Network(object):
    """
    A Network class that provides an interface of socket communication between flower and
    Ns3 Network Simulator.

    Methods:
        start_ns3(visualize): Starts the Ns3 Network Simulator.
        parse_clients(clients): Parse the clients into binary array format depicting the training participants.
        connect(): Try to establish socket connection with Ns3.
        sendRequest(requestType, client_array): Send a synchronous round request and receive a response about Ns3 calculated round stats.
        sendAsyncRequest(requestType, client_array): Send an asynchronous round request.
        readAsyncResoponse(): Read a response from Ns3 about an asynchronous round request.
        disconnect(): Disconnect from Ns3 closing connection.
    """

    def __init__(self, config: Dict[str, Any], client_cfg: Dict[str, Any]):
        # Class variables extracted from config
        self.tcp_ip = config.tcp_ip
        self.wifi_net_template = config.wifi_net_template
        self.port = config.port
        self.path = config.path
        self.program = config.program
        self.num_clients = client_cfg.total
        self.clients_for_fit = client_cfg.for_fit
        self.network_type = config.network_type
        self.server_type = config.server
        self.dataset = config.dataset
        self.ann = config.ann
        self.moving_clients = config.moving_clients

        self.device_type = config.device_type
        self.net_cfg = (
            config.network.wifi
            if self.network_type == "wifi"
            else self.config.network.ethernet
        )

        if self.dataset not in config.model_sizes:
            raise ValueError(
                f"Error Server Config for Dataset {self.dataset} and ANN: {self.ann}"
            )

        model_sizes = config.model_sizes[self.dataset]
        self.model_size = model_sizes.get(self.ann, model_sizes["default"])
        print(f"Model size for {self.dataset} with {self.ann}: {self.model_size}")

    # =====================================================================================================================
    # * Utility functions *

    def start_ns3(self, visualize=False):
        """
        Start running the ns3 app. Build at first place and run afterwards.

        Parameters:
            visualize (bool): Option to run pyVis in parallel with simulation to visualize (not recommended. Better to use NetAnim).
        """
        # Assuming the script is in the project's home directory
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        # Change the working directory
        os.chdir(cur_dir)

        # Confirm the current working directory
        print(f"\nCurrent Working Directory: {cur_dir}")

        print("==================== NS3 Network =================")
        if not self._is_configured():
            print("Project configure")
            proc = subprocess.Popen(
                "./waf configure --build-profile=optimized",
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

        # print("Wait For Buld to Complete. Might take a while\n")
        # proc = subprocess.Popen(
        #     "./waf build",
        #     shell=True,
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     universal_newlines=True,
        #     cwd=self.path,
        # )
        # # Read and print output and errors
        # while True:
        #     output = proc.stdout.readline()
        #     if output == "" and proc.poll() is not None:
        #         break
        #     if output:
        #         print(output.strip())

        # rc = proc.poll()
        # print(f"Process exited with code {rc}")

        # proc.wait()

        # if proc.returncode != 0:
        #     print("A problem occured while building")
        #     exit(-1)

        # print("Build complete!\n\n\n", flush=True)
        moving_clients_str = "true" if self.moving_clients else "false"
        command = (
            './waf --run "'
            + self.program
            + " --numClients="
            + str(self.num_clients)
            + " --networkType="
            + self.network_type
            + " --movingClients="
            + str(moving_clients_str)
            + " --modelSize="
            + str(self.model_size)
            + " --wifiNetTemplate="
            + str(self.wifi_net_template)
        )

        """print(self.config.network)
        for net in self.config.network:
            if net == self.network_type:
                print(net.items())"""

        if self.network_type == "wifi":
            command += " --txGain=" + str(self.net_cfg.tx_gain)
            command += " --maxPacketSize=" + str(self.net_cfg.max_packet_size)
        else:  # else assume ethernet
            command += " --maxPacketSize=" + str(self.net_cfg.max_packet_size)

        command += (
            " --learningModel="
            + str(self.server_type)
            + " --deviceType="
            + str(self.device_type)
            + " --modelType="
            + str(self.dataset)
        )

        command += '" --vis ' if visualize else '"'
        title = "NS3"
        print(f"Command to execute: {command}")

        terminal_command = f"screen -dmS NS3 bash -c '{command}; echo \"Press any key to close...\"; read -n 1'"

        # terminal_command = f"gnome-terminal --title={title} -- bash -c '{command}; echo \"Press any key to close...\"; read -n 1'"

        proc = subprocess.Popen(
            terminal_command,
            shell=True,
            cwd=self.path,
        )

    def _is_configured(self) -> bool:
        """
        Check if waf is configured.
        """
        # Path to a file or directory that indicates configuration is done
        config_marker = os.path.join(self.path, "build/config.log")

        # Only run configuration if the marker doesn't exist
        if not os.path.exists(config_marker):
            return False
        return True

    def parse_clients(self, clients: List) -> List:
        """
        Parse the clients into binary array format depicting the training participants.

        Parameters:
            clients: list of clients to be used in training. Assuming 4 total client with 3 participating in fit.
            Initial format: [1,0,3] -> Parsed: [1,1,0,1].

        Returns:
            The Parsed client list.
        """
        clients_to_send = [0 for _ in range(self.num_clients)]
        for index in clients:
            clients_to_send[index] = 1
        return clients_to_send

    # =====================================================================================================================
    # * Communication Interface Methods *
    def connect(self):
        try:
            print(f"Attempting to connect to NS3 at {self.tcp_ip}:{self.port}")
            self.socket = socket.create_connection((self.tcp_ip, self.port))
            print(f"Successfully connected to NS3 at {self.tcp_ip}:{self.port}")
        except ConnectionRefusedError as e:
            print(f"ConnectionRefusedError: Failed to connect to NS3 at {self.tcp_ip}:{self.port} - {e}")
            raise
        except Exception as e:
            print(f"Unexpected error occurred while connecting to NS3 at {self.tcp_ip}:{self.port} - {e}")
            raise

    def sendRequest(self, requestType: int, array: list) -> Dict[int, Dict[str, float]]:
        """
        Send a synchronous round request and receive a response about Ns3 calculated round stats.

        Parameters:
            requestType (int): The type of request: [0->Response, 1->StartSim, 2->Exit, 3->EndSim(for async)].
            array (list): The parsed array of clients used for training.

        Returns:
            The calculated statistics used in training: {id: {"downLinkTime": float, "upinkTime": float, "throughput": float}}
        """
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

    def sendAsyncRequest(self, requestType: int, array: list):
        """
        Send an asynchronous round request

        Parameters:
            requestType (int): The type of request: [0->Response, 1->StartSim, 2->Exit, 3->EndSim(for async)].
            array (list): The parsed array of clients used for training.
        """
        print("sending")
        print(array)
        message = struct.pack("II", requestType, len(array))
        self.socket.send(message)
        # for the total number of clients
        # is the index in lit at client.id equal
        for ele in array:
            self.socket.send(struct.pack("I", ele))

    def readAsyncResponse(self) -> Dict[int, Dict[str, float]]:
        """
        Receive a response about Ns3 calculated round stats in an asynchronous round.

        Returns:
            The calculated statistics used in training: {id: {"startTime": float, "endTime": float, "throughput": float}}
        """
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
        """
        Disconnect from Ns3 closing connection.
        """
        # self.socketendAsyncRequest(requestType=2, array=[])
        self.socket.close()
