# third-party
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

# local
from utils.monitor import Monitor


class CustomClientManager(SimpleClientManager):
    """
    @extends flwr.server.client_manager.SimpleClientManager
    This ia a slightly modified version of the flower client manager
    with the subtle modification of preserving a mapping of client
    proxies cids to arithmetic ids(ex. 0,1 etc).
    """

    def __init__(self, monitor: Monitor):
        super().__init__()
        self.monitor = monitor

    def register(self, client: ClientProxy):
        """
        @override
        If successful, registration add cid of proxy to the cid-to-proxy mapping.
        For more information refer to the flwr.server.client_manager.SimpleClientManager documentation.

        Returns:
            Boolean: True if registration was successful
        """
        succ = super().register(client)

        if succ:
            self.monitor.update_id_mappings(client.cid)
        return succ
