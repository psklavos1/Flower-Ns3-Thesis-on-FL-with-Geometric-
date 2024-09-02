import os
import pandas as pd
import logging


class DataHandler:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.client_batch_metrics = {}
        self.client_epoch_results = {}
        self.server_aggregated_results_df = pd.DataFrame()
        self.experiment_info_df = pd.DataFrame()
        self.tmp_df = pd.DataFrame()

    def _get_df(self, level, client_id=None):
        """Helper function to get or initialize the DataFrame for a specific level and client."""
        if level == "batch":
            if client_id not in self.client_batch_metrics:
                self.client_batch_metrics[client_id] = pd.DataFrame()
            return self.client_batch_metrics[client_id]
        elif level == "epoch":
            if client_id not in self.client_epoch_results:
                self.client_epoch_results[client_id] = pd.DataFrame()
            return self.client_epoch_results[client_id]
        elif level == "server":
            return self.server_aggregated_results_df
        elif level == "info":
            return self.experiment_info_df
        elif level == "tmp":
            return self.tmp_df

    def _set_df(self, df, level, client_id=None):
        """Helper function to set the DataFrame for a specific client and level."""
        if level == "batch":
            self.client_batch_metrics[client_id] = df
        elif level == "epoch":
            self.client_epoch_results[client_id] = df
        elif level == "server":
            self.server_aggregated_results_df = df
        elif level == "info":
            self.experiment_info_df = df
        elif level == "tmp":
            self.tmp_df = df

    def add_data(self, data: dict, level: str, client_id: str = None):
        """Adds data to the appropriate DataFrame (batch, epoch, server, info)."""
        if level not in ["batch", "epoch", "server", "info", "tmp"]:
            raise ValueError(
                "Level must be 'batch', 'epoch', 'server', 'tmp' or 'info'"
            )
        data_df = pd.DataFrame([data])
        df = self._get_df(level, client_id)
        df = pd.concat([df, data_df], ignore_index=True)
        self._set_df(df, level, client_id)

    def append_data_to_last_row(
        self, column: str, value, level: str, client_id: str = None
    ):
        """Appends data to the specified column in the last row of the DataFrame."""
        if level not in ["batch", "epoch", "server", "info", "tmp"]:
            raise ValueError(
                "Level must be 'batch', 'epoch', 'server', 'tmp' or 'info'"
            )

        df = self._get_df(level, client_id)
        if not df.empty:
            df.at[df.index[-1], column] = value
            self._set_df(df, level, client_id)
        else:
            logging.warning(
                f"No data to append to in {level} data for client {client_id}"
            )

    def append_data_to_column(
        self, column: str, values: list, level: str, client_id: str = None
    ):
        """Appends data to the specified column in the DataFrame."""
        if level not in ["batch", "epoch", "server", "info", "tmp"]:
            raise ValueError(
                "Level must be 'batch', 'epoch', 'server', 'tmp' or 'info'"
            )

        df = self._get_df(level, client_id)
        if not df.empty:
            for i, value in enumerate(values):
                if i < len(df):
                    df.at[df.index[i], column] = value
                else:
                    new_row = {col: None for col in df.columns}
                    new_row[column] = value
                    df = df.append(new_row, ignore_index=True)
            self._set_df(df, level, client_id)

    def clear_data(self, level: str, client_id: str = None):
        """Clears the contents of the specified DataFrame."""
        if level not in ["batch", "epoch", "server", "info", "tmp"]:
            raise ValueError(
                "Level must be 'batch', 'epoch', 'server', 'tmp' or 'info'"
            )

        self._set_df(pd.DataFrame(), level, client_id)

    def get_data(self, level: str, client_id: str = None, clear_after= False):
        """Returns the data for the specified level."""
        if level not in ["batch", "epoch", "server", "info", "tmp"]:
            raise ValueError(
                "Level must be 'batch', 'epoch', 'server', 'tmp' or 'info'"
            )
        if (level == "batch" or level == "epoch") and not client_id:
            raise ValueError("Client_id must be specified.")
        ret = self._get_df(level, client_id)
        if clear_after:
            self.clear_data(level, client_id)
        return ret

    def find_data(self, column: str, value, level: str, client_id: str = None):
        """Finds data by column value."""
        df = self.get_data(level, client_id)

        return df[df[column] == value]

    def save_data(
        self, filename: str, level: str, directory: str, client_id: str = None
    ):
        """Saves the data to a CSV file."""
        df = self.get_data (level, client_id)
        # if level == "server":
        #     print(df)
        
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        filepath = os.path.join(directory, filename)
        try:
            df.to_csv(filepath, index=False)
            logging.info(f"{level.capitalize()} data saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save {level} data: {e}")

    def load_data(
        self, filename: str, level: str, directory: str, client_id: str = None
    ):
        """Loads data from a CSV file."""
        if level not in ["batch", "epoch", "server", "info", "tmp"]:
            raise ValueError(
                "Level must be 'batch', 'epoch', 'server', 'tmp' or 'info'"
            )

        filepath = os.path.join(directory, filename)
        try:
            df = pd.read_csv(filepath)
            self._set_df(df, level, client_id)
            logging.info(f"{level.capitalize()} data loaded from {filepath}")
        except Exception as e:
            logging.error(f"Failed to load {level} data: {e}")

    def delete_file(self, filename: str, directory: str):
        """Deletes a file."""
        filepath = os.path.join(directory, filename)
        try:
            os.remove(filepath)
            logging.info(f"File {filepath} deleted")
        except Exception as e:
            logging.error(f"Failed to delete file {filepath}: {e}")

    def aggregate_info_files(self, root_dir: str):
        all_info = pd.DataFrame()

        # Iterate over all directories in the root directory
        for exp_id in sorted(os.listdir(root_dir)):
            exp_path = os.path.join(root_dir, exp_id)
            if os.path.isdir(exp_path):
                info_file = os.path.join(exp_path, "info.csv")
                if os.path.exists(info_file):
                    # Read the info.csv file
                    info_df = pd.read_csv(info_file)
                    # Ensure the exp_id is included only once
                    if "exp_id" not in info_df.columns:
                        info_df["exp_id"] = exp_id
                    # Append to the overall dataframe
                    all_info = pd.concat([all_info, info_df], ignore_index=True)
        return all_info
