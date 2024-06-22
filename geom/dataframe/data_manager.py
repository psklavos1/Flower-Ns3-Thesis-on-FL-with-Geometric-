import logging
from dataframe.data_handler import DataHandler
from dataframe.file_manager import FileManager
from dataframe.data_plotter import DataPlotter


class DataManager:
    def __init__(self, log_dir="csv_logs"):
        # logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.log_dir = log_dir

        # Handlers
        self.file_manager = FileManager(log_dir)
        self.data_handler = DataHandler()
        self.data_plotter = DataPlotter(self.data_handler)

        self.experiment_id = self.file_manager.get_experiment_id()
        self.experiment_dir = self.file_manager.get_experiment_dir()
        logging.info(
            f"Initialized DataManager for experiment {self.file_manager.get_experiment_id()}"
        )

    # * ================================ Add Data ================================
    def add_batch_data(self, new_data: dict, client_id: str, log=False):
        """Adds batch-level data to the DataHandler."""
        new_data["experiment_id"] = self.experiment_id
        new_data["client_id"] = client_id
        self.data_handler.add_data(new_data, level="batch", client_id=client_id)
        if log:
            logging.info(f"Added batch data for client {client_id}: {new_data}")

    def add_batch_data_list(self, new_data: list, client_id: str, log=False):
        """Adds batch-level data to the DataHandler."""
        for data in new_data:
            data["experiment_id"] = self.experiment_id
            data["client_id"] = client_id
            self.data_handler.add_data(data, level="batch", client_id=client_id)
        if log:
            logging.info(
                f"Added batch data list for client {client_id}. New Data Added: {len(new_data)}"
            )

    def add_epoch_data(self, new_data: dict, client_id: str, log=False):
        """Adds epoch-level data to the DataHandler."""
        new_data["experiment_id"] = self.experiment_id
        new_data["client_id"] = client_id
        self.data_handler.add_data(new_data, level="epoch", client_id=client_id)
        if log:
            input()
            logging.info(f"Added epoch data for client {client_id}: {new_data}")

    def add_server_data(self, new_data: dict, log=False):
        """Adds server-level data to the DataHandler."""
        new_data["experiment_id"] = self.experiment_id
        self.data_handler.add_data(new_data, level="server")
        if log:
            logging.info(f"Added server data: {new_data}")

    def add_info_data(self, new_data: dict, log=False):
        """Adds info-level data to the DataHandler."""
        data = {"experiment_id": self.experiment_id}
        data.update(new_data)
        self.data_handler.add_data(data, level="info")
        if log:
            logging.info(f"Added info data: {data}")

    # * ================================ Append Data Row ================================
    def append_batch_data_to_last_row(self, data: dict, client_id: str, log=False):
        """Appends data to the specified columns in the last row of the batch DataFrame."""
        if self.data_handler.get_data("batch", client_id=client_id).empty:
            self.add_batch_data(data, client_id=client_id)
        else:
            for column, value in data.items():
                self.data_handler.append_data_to_last_row(
                    column, value, level="batch", client_id=client_id
                )
        if log:
            logging.info(
                f"Appended {data} to the last row of batch data for client {client_id}"
            )

    def append_epoch_data_to_last_row(self, data: dict, client_id: str, log=False):
        """Appends data to the specified columns in the last row of the epoch DataFrame."""
        if self.data_handler.get_data("epoch", client_id=client_id).empty:
            self.add_epoch_data(data, client_id=client_id)
        else:
            for column, value in data.items():
                self.data_handler.append_data_to_last_row(
                    column, value, level="epoch", client_id=client_id
                )
        if log:
            logging.info(
                f"Appended {data} to the last row of epoch data for client {client_id}"
            )

    def append_server_data_to_last_row(self, data: dict, log=False):
        """Appends data to the specified columns in the last row of the server DataFrame."""
        if self.data_handler.get_data("server").empty:
            self.add_server_data(data)
        else:
            for column, value in data.items():
                self.data_handler.append_data_to_last_row(column, value, level="server")
        if log:
            logging.info(f"Appended {data} to the last row of server data")

    def append_info_data_to_last_row(self, data: dict, log=False):
        """Appends data to the specified columns in the last row of the info DataFrame."""
        if self.data_handler.get_data("info").empty:
            self.add_info_data(data)
        else:
            for column, value in data.items():
                self.data_handler.append_data_to_last_row(column, value, level="info")
        if log:
            logging.info(f"Appended {data} to the last row of info data")

    # * ================================ Append Data Col ================================
    def append_batch_data_to_column(
        self, column: str, values: list, client_id: str, log=False
    ):
        """Appends data to the specified column in the batch DataFrame."""
        self.data_handler.append_data_to_column(
            column, values, level="batch", client_id=client_id
        )
        if log:
            logging.info(
                f"Appended {values} to column {column} in batch data for client {client_id}"
            )

    def append_epoch_data_to_column(
        self, column: str, values: list, client_id: str, log=False
    ):
        """Appends data to the specified column in the epoch DataFrame."""
        self.data_handler.append_data_to_column(
            column, values, level="epoch", client_id=client_id
        )
        if log:
            logging.info(
                f"Appended {values} to column {column} in epoch data for client {client_id}"
            )

    def append_server_data_to_column(self, column: str, values: list, log=False):
        """Appends data to the specified column in the server DataFrame."""
        self.data_handler.append_data_to_column(column, values, level="server")
        if log:
            logging.info(f"Appended {values} to column {column} in server data")

    def append_info_data_to_column(self, column: str, values: list, log=False):
        """Appends data to the specified column in the info DataFrame."""
        self.data_handler.append_data_to_column(column, values, level="info")
        if log:
            logging.info(f"Appended {values} to column {column} in info data")

    # * ================================ Save Data ================================
    def save_batch_data(self, filename="batch", client_id=""):
        """Saves batch-level data to a CSV file."""
        filename += ".csv"
        in_exp_dir = "/clients/client_" + client_id + "/"
        dir = self.experiment_dir + in_exp_dir
        self.data_handler.save_data(
            filename, level="batch", directory=dir, client_id=client_id
        )

    def save_epoch_data(self, filename="epoch", client_id=""):
        """Saves epoch-level data to a CSV file."""
        filename += ".csv"
        in_exp_dir = "/clients/client_" + client_id + "/"
        dir = self.experiment_dir + in_exp_dir
        self.data_handler.save_data(
            filename, level="epoch", directory=dir, client_id=client_id
        )

    def save_server_data(self, filename="server"):
        """Saves server-level data to a CSV file."""
        filename += ".csv"
        dir = self.experiment_dir + "/server"
        self.data_handler.save_data(filename, level="server", directory=dir)

    def save_info_data(self, filename="info", dir=""):
        """Saves info-level data to a CSV file."""
        filename += ".csv"
        self.data_handler.save_data(
            filename, level="info", directory=self.experiment_dir
        )

    def save_all_data(
        self,
        batch_filename,
        epoch_filename,
        server_filename,
        info_filename,
        client_id="",
    ):
        """Saves all data to CSV files."""
        self.save_batch_data(batch_filename, client_id=client_id)
        self.save_epoch_data(epoch_filename, client_id=client_id)
        self.save_server_data(server_filename)
        self.save_info_data(info_filename)

    # * ================================ Search Data ================================
    def find_batch_data(self, column: str, value, client_id: str):
        """Finds batch-level data by column value."""
        result = self.data_handler.find_data(
            column, value, level="batch", client_id=client_id
        )
        logging.info(
            f"Found batch data for client {client_id} where {column} = {value}"
        )
        return result

    def find_epoch_data(self, column: str, value, client_id: str):
        """Finds epoch-level data by column value."""
        result = self.data_handler.find_data(
            column, value, level="epoch", client_id=client_id
        )
        logging.info(
            f"Found epoch data for client {client_id} where {column} = {value}"
        )
        return result

    def find_server_data(self, column: str, value):
        """Finds server-level data by column value."""
        result = self.data_handler.find_data(column, value, level="server")
        logging.info(f"Found server data where {column} = {value}")
        return result

    def find_info_data(self, column: str, value):
        """Finds info-level data by column value."""
        result = self.data_handler.find_data(column, value, level="info")
        logging.info(f"Found info data where {column} = {value}")
        return result

    # * ================================ Plot Data ================================
    def plot_batch_data(
        self,
        plot_type: str,
        x: str,
        y: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        client_id: str = None,
    ):
        """Plots batch-level data."""
        logging.info(
            f"Plotting batch data: {plot_type} plot of {y} vs {x} for client {client_id}"
        )
        self.data_plotter.plot_data(
            plot_type,
            x,
            y,
            level="batch",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            client_id=client_id,
        )

    def plot_epoch_data(
        self,
        plot_type: str,
        x: str,
        y: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        client_id: str = None,
    ):
        """Plots epoch-level data."""
        logging.info(
            f"Plotting epoch data: {plot_type} plot of {y} vs {x} for client {client_id}"
        )
        self.data_plotter.plot_data(
            plot_type,
            x,
            y,
            level="epoch",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            client_id=client_id,
        )

    def plot_server_data(
        self,
        plot_type: str,
        x: str,
        y: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        """Plots server-level data."""
        logging.info(f"Plotting server data: {plot_type} plot of {y} vs {x}")
        self.data_plotter.plot_data(
            plot_type, x, y, level="server", title=title, xlabel=xlabel, ylabel=ylabel
        )

    def plot_info_data(
        self,
        plot_type: str,
        x: str,
        y: str,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
    ):
        """Plots info-level data."""
        logging.info(f"Plotting info data: {plot_type} plot of {y} vs {x}")
        self.data_plotter.plot_data(
            plot_type, x, y, level="info", title=title, xlabel=xlabel, ylabel=ylabel
        )
