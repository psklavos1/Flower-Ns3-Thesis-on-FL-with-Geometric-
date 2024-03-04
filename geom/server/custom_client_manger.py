# custom_client_manager.py
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from utils.monitor import Monitor


class CustomClientManager(SimpleClientManager):
    def __init__(self, monitor: Monitor):
        super().__init__()
        self.monitor = monitor

    def register(self, client: ClientProxy) -> None:
        succ = super().register(client)

        if succ:
            self.monitor.update_id_mappings(client.cid)
        return succ

    
