class Monitor:
    """Keeps track of various round variables"""

    def __init__(self) -> None:
        self.num_fit_clients = 0
        self.num_eval_clients = 0
        self.round = 0

        # Mapping for Ns3 Compatibility
        self.last_id = 0
        self.cid_to_index = {}
        self.index_to_cid = {}

    def reset(
        self,
    ):
        self.num_fit_clients = 0
        self.num_eval_clients = 0
        self.round = 0

        return

    # Getters
    def get_fit_clients(self):
        return self.num_fit_clients

    def get_eval_clients(self):
        return self.num_eval_clients

    def get_round(self):
        return self.round

    def get_cid_to_index_map(self):
        return self.cid_to_index

    def get_index_to_cid_map(self):
        return self.index_to_cid

    def get_index(self, cid: str):
        return self.cid_to_index.get(cid)

    def get_cid(self, index: int):
        return self.index_to_cid.get(index)

    # Setters
    def set_fit_clients(self, x):
        self.num_fit_clients = x

    def set_eval_clients(self, x):
        self.num_eval_clients = x

    def set_round(self, x):
        self.round = x

    def set_cid_to_index(self, x):
        self.cid_to_index = x

    def set_index_to_cid(self, x):
        self.index_to_cid = x

    def update_id_mappings(self, cid):
        self.cid_to_index[cid] = self.last_id
        self.index_to_cid[self.last_id] = cid
        self.last_id += 1
