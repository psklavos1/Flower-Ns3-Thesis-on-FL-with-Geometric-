class Monitor:
    """
    A utility class to keep track of various variables across rounds in a federated learning scenario or similar setups.

    This class tracks the number of clients participating in fitting and evaluation, the current round of operation,
    and mappings between client IDs and indices for compatibility with systems like NS3.

    Attributes:
        num_fit_clients (int): Number of clients participating in model fitting.
        num_eval_clients (int): Number of clients participating in model evaluation.
        round (int): The current round of operation.
        last_id (int): The last used numerical ID for mapping new client IDs.
        cid_to_index (dict): A mapping from client IDs to numerical indices.
        index_to_cid (dict): A mapping from numerical indices to client IDs.
    """

    def __init__(self) -> None:
        self.num_fit_clients = 0
        self.num_eval_clients = 0
        self.round = 0

        # Mapping for Ns3 Compatibility
        self.last_id = 0
        self.cid_to_index = {}
        self.index_to_cid = {}

    def reset(self):
        """
        Resets the monitor's tracking variables to their initial state.
        """
        self.num_fit_clients = 0
        self.num_eval_clients = 0
        self.round = 0

    # Getters
    def get_fit_clients(self):
        """
        Returns the number of fit clients.

        Returns:
            int: The number of clients participating in model fitting.
        """
        return self.num_fit_clients

    def get_eval_clients(self):
        """
        Returns the number of evaluation clients.

        Returns:
            int: The number of clients participating in model evaluation.
        """
        return self.num_eval_clients

    def get_round(self):
        """
        Returns the current round.

        Returns:
            int: The current operation round.
        """
        return self.round

    def get_cid_to_index_map(self):
        """
        Returns the mapping of client IDs to numerical indices.

        Returns:
            dict: The client ID to index mapping.
        """
        return self.cid_to_index

    def get_index_to_cid_map(self):
        """
        Returns the mapping of numerical indices to client IDs.

        Returns:
            dict: The index to client ID mapping.
        """
        return self.index_to_cid

    def get_index(self, cid: str):
        """
        Retrieves the numerical index associated with a given client ID.

        Args:
            cid (str): The client ID.

        Returns:
            int: The associated numerical index, or None if not found.
        """
        return self.cid_to_index.get(cid)

    def get_cid(self, index: int):
        """
        Retrieves the client ID associated with a given numerical index.

        Args:
            index (int): The numerical index.

        Returns:
            str: The associated client ID, or None if not found.
        """
        return self.index_to_cid.get(index)

    # Setters
    def set_fit_clients(self, x: int):
        """
        Sets the number of fit clients.

        Args:
            x (int): The number of clients participating in model fitting.
        """
        self.num_fit_clients = x

    def set_eval_clients(self, x: int):
        """
        Sets the number of evaluation clients.

        Args:
            x (int): The number of clients participating in model evaluation.
        """
        self.num_eval_clients = x

    def set_round(self, x: int):
        """
        Sets the current round.

        Args:
            x (int): The current operation round.
        """
        self.round = x

    def update_id_mappings(self, cid: str):
        """
        Updates the mappings of client IDs to indices and vice versa with a new client ID.

        This method assigns a new unique numerical index to the provided client ID and updates
        the internal mappings accordingly.

        Args:
            cid (str): The client ID to add to the mappings.
        """
        self.cid_to_index[cid] = self.last_id
        self.index_to_cid[self.last_id] = cid
        self.last_id += 1
