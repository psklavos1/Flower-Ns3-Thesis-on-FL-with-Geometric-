import os
from datetime import datetime


class FileManager:
    def __init__(self, log_dir="logs_dir"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.experiment_dir, self.experiment_id = self._catalog_experiment()

    def _catalog_experiment(self):
        """Initializes a new experiment by creating a unique directory for its logs."""
        experiment_id = self._generate_experiment_id()
        experiment_dir = os.path.join(self.log_dir, experiment_id)

        # Create clients and server subdirectories
        clients_dir = os.path.join(experiment_dir, "clients")
        server_dir = os.path.join(experiment_dir, "server")

        os.makedirs(clients_dir, exist_ok=True)
        os.makedirs(server_dir, exist_ok=True)

        return experiment_dir, experiment_id

    def _generate_experiment_id(self):
        """Generates a unique experiment ID based on the current timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"

    def get_experiment_dir(self):
        """Returns the full path of the current experiment log directory."""
        return self.experiment_dir

    def get_experiment_id(self):
        """Returns the current experiment ID."""
        return self.experiment_id
