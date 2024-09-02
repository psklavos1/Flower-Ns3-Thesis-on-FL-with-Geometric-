import logging
import argparse
from dataframe.data_handler import DataHandler
from dataframe.data_plotter import DataPlotter
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
ROOT_DIRECTORY = "../csv_logs/results"


def get_column_sum(df: pd.DataFrame, column_name: str, limit: int = None) -> float:
    """
    Prints and returns the sum of the specified column in the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column for which to calculate the sum.
        limit (int, optional): The number of rows to consider for the calculation.

    Returns:
        float: The sum of the specified column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    if limit is not None:
        df = df.head(limit)

    column_sum = df[column_name].sum()

    print(f"Sum of {column_name}: {column_sum:.4f}")

    return column_sum


def get_column_max(df: pd.DataFrame, column_name: str, limit: int = None) -> float:
    """
    Prints and returns the peak (maximum) value of the specified column in the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column for which to get the peak value.
        label (str, optional): An optional label to use when printing the peak value.
        limit (int, optional): The number of rows to consider for the calculation.

    Returns:
        float: The peak (maximum) value of the specified column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    if limit is not None:
        df = df.head(limit)

    peak_value = df[column_name].max()

    print(f"Peak (max) of {column_name}: {peak_value:.4f}")

    return peak_value


def get_column_average(df: pd.DataFrame, column_name: str, limit: int = None) -> float:
    """
    Prints and returns the average of the specified column in the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column for which to calculate the average.
        label (str, optional): An optional label to use when printing the average.

    Returns:
        float: The average of the specified column.
    """
    df = df.iloc[1:]
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    if limit is not None:
        df = df.head(limit)
    avg_value = df[column_name].mean()

    print(f"Average of {column_name}: {avg_value:.2f}")

    return avg_value


def get_non_iid(directory, data_handler: DataHandler):

    logging.info(f"Loading info.csv from {directory}")
    data_handler.load_data(filename="info.csv", directory=directory, level="info")

    info_df = data_handler.get_data("info", clear_after=True)

    if info_df is None or info_df.empty:
        logging.error("Info data is not available or empty")
        raise ValueError("Info data is not available or empty")

    logging.info(f"Columns in loaded info_df: {info_df.columns.tolist()}")

    if "non_iid" not in info_df.columns:
        logging.error("The 'non_iid' column is not present in the info_df")
        raise KeyError("The 'non_iid' column is not present in the info_df")

    non_iid = info_df["non_iid"].values[0]
    bias_template = (
        info_df["bias_template"].values[0]
        if "bias_template" in info_df.columns
        else None
    )
    return non_iid, bias_template


def get_client_ids(exp_id, root_directory):
    client_ids = []
    clients_path = os.path.join(root_directory, exp_id, "clients")
    if os.path.exists(clients_path):
        for entry in os.listdir(clients_path):
            if os.path.isdir(os.path.join(clients_path, entry)) and entry.startswith(
                "client_"
            ):
                client_id = entry.split("_")[1]
                client_ids.append(client_id)
    return client_ids


def get_total_steps(root_directory, exp_id, limit: int = None):
    clients_path = os.path.join(root_directory, exp_id, "clients")
    for client_id in os.listdir(clients_path):
        client_epoch_path = os.path.join(clients_path, client_id, "epoch.csv")
        if os.path.exists(client_epoch_path):
            epoch_df = pd.read_csv(client_epoch_path)
            total_steps = (
                epoch_df["total_steps"].iloc[-1]
                if not limit
                else epoch_df["total_steps"].iloc[limit]
            )
            return total_steps
    raise FileNotFoundError(
        f"No epoch.csv file found in any client directory of {exp_id}"
    )


def get_val_acc(root_directory: str, exp_id: str, first=True):
    clients_path = os.path.join(root_directory, exp_id, "clients")
    for client_id in os.listdir(clients_path):
        if not first:
            first = True
            continue
        client_epoch_path = os.path.join(clients_path, client_id, "epoch.csv")
        if os.path.exists(client_epoch_path):
            epoch_df = pd.read_csv(client_epoch_path)
            return epoch_df[["round", "val_accuracy"]]
    raise FileNotFoundError(
        f"No epoch.csv file found in any client directory of {exp_id}"
    )


def get_info_data(directory):
    info_path = os.path.join(directory, "info.csv")
    if os.path.exists(info_path):
        info_df = pd.read_csv(info_path)
        return info_df
    else:
        raise FileNotFoundError(f"No such file: {info_path}")


def prepare_pie_chart_data(
    data_handler: DataHandler, experiment_id: str, root_directory: str
):
    # Load info data
    experiment_directory = os.path.join(root_directory, experiment_id)
    info_df = get_info_data(experiment_directory)
    algorithm = info_df["algorithm"].values[0]

    # Load server data
    server_directory = os.path.join(experiment_directory, "server")
    data_handler.load_data(
        filename="server.csv",
        level="server",
        directory=server_directory,
    )

    server_data = data_handler.get_data("server", clear_after=True)
    server_data.iloc[1:]

    avg_downlink_time = server_data["average_downlink_time"].mean()
    avg_rtc_check_time = (
        server_data["average_rtc_check_time"].mean() if algorithm == "fda" else None
    )
    avg_uplink_time = server_data["average_uplink_time"].mean()

    if algorithm == "fda":
        pie_data = pd.DataFrame(
            {"Time": [avg_downlink_time, avg_rtc_check_time, avg_uplink_time]},
            index=[
                "Average Downlink Time",
                "Average RTC Check Time",
                "Average Uplink Time",
            ],
        )
    else:
        pie_data = pd.DataFrame(
            {"Time": [avg_downlink_time, avg_uplink_time]},
            index=[
                "Average Downlink Time",
                "Average Uplink Time",
            ],
        )

    return pie_data


def plot_data_percentage(
    data_handler: DataHandler, plotter: DataPlotter, bias_template: int
):
    # Initialize the dictionary with classes and percentages
    classes = [
        "Class 0",
        "Class 1",
        "Class 2",
        "Class 3",
        "Class 4",
        "Class 5",
        "Class 6",
        "Class 7",
        "Class 8",
        "Class 9",
    ]
    percentages = {
        1: [11.78, 13.75, 8.33, 7.12, 9.99, 8.62, 9.44, 12.71, 9.45, 8.81],
        2: [11.85, 5.08, 7.60, 5.53, 5.50, 11.27, 13.42, 28.30, 4.97, 6.46],
        3: [7.19, 5.10, 9.86, 2.85, 5.12, 37.71, 7.55, 12.44, 3.75, 8.42],
    }
    percentage_values = percentages.get(bias_template, [0] * 10)

    # Add each row separately
    for class_name, percentage in zip(classes, percentage_values):
        data = {"class": class_name, "percentage": percentage}
        data_handler.add_data(data=data, level="tmp")

    # Print the contents of the 'tmp' DataFrame
    tmp_df = data_handler.get_data(level="tmp", clear_after=True)
    print(tmp_df)

    plotter.plot_data(
        plot_type="bar",
        x="class",
        y=["percentage"],
        level="tmp",
        title="Class percentages",
        xlabel="Class",
        ylabel="Percentage",
    )


def plot_accuracy_vs_round(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    group_by: str,
    comment: str = "",
    limit: int = None,
    centralized=False,
):
    combined_data = pd.DataFrame()
    filename = "server.csv"
    dir = "server/"

    # Requires first the fda
    fda_steps: int = None
    for exp_id in experiment_ids:
        experiment_directory = os.path.join(root_directory, exp_id)
        file_directory = os.path.join(experiment_directory, dir)
        logging.info(f"Loading data for experiment ID: {exp_id}")
        data_handler.load_data(
            filename=filename,
            level="server",
            directory=file_directory,
        )

        exp_data = data_handler.get_data("server", clear_after=True)
        info_df = get_info_data(experiment_directory)
        if pd.isna(info_df[group_by].values[0]) or info_df[group_by].values[0] == "":
            exp_data[group_by] = "Default"
        else:
            exp_data[group_by] = info_df[group_by].values[0]

        print(info_df[group_by].values[0])
        algorithm = info_df["algorithm"].values[0]

        if algorithm == "fda":
            if limit is not None:
                exp_data = exp_data.head(limit + 1)
                fda_steps = get_total_steps(root_directory, exp_id, limit + 1)
            else:
                fda_steps = get_total_steps(root_directory, exp_id)
        elif algorithm == "synchronous":
            if limit is not None and fda_steps is not None:
                limit = fda_steps
                exp_data = exp_data.head(limit)
        combined_data = pd.concat([combined_data, exp_data])
    # comment = f"Batches processed: {fda_steps}"

    y_print = "distributed_acc" if not centralized else "centralized_acc"
    y_label = "Distributed" if not centralized else "Centralized"
    plotter.plot_combined_data(
        plot_type="line",
        x="round",
        y=y_print,
        data=combined_data,
        group_by=group_by,
        title=f"{y_label} Accuracy over Rounds",
        xlabel="Round",
        ylabel=f"{y_label} Accuracy",
        comment=comment,
    )


def plot_accuracy_vs_training_time(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    group_by: str,
    comment: str = "",
    limit: int = None,
):
    combined_data = pd.DataFrame()
    filename = "server.csv"
    dir = "server/"

    # Requires first the fda
    fda_steps: int = None
    print(limit)
    for exp_id in experiment_ids:
        experiment_directory = os.path.join(root_directory, exp_id)
        file_directory = os.path.join(experiment_directory, dir)
        logging.info(f"Loading data for experiment ID: {exp_id}")
        data_handler.load_data(
            filename=filename,
            level="server",
            directory=file_directory,
        )

        exp_data = data_handler.get_data("server", clear_after=True)
        info_df = get_info_data(experiment_directory)
        if pd.isna(info_df[group_by].values[0]) or info_df[group_by].values[0] == "":
            exp_data[group_by] = "Default"
        else:
            exp_data[group_by] = info_df[group_by].values[0]

        algorithm = info_df["algorithm"].values[0]

        # Adjust round time based on algorithm type
        if algorithm == "fda":
            exp_data["total_round_time"] = exp_data["round_time"]
            if limit is not None:
                exp_data = exp_data.head(limit + 1)
                fda_steps = get_total_steps(root_directory, exp_id, limit + 1)
            else:
                fda_steps = get_total_steps(root_directory, exp_id)
        elif algorithm == "synchronous":
            exp_data["total_round_time"] = (
                exp_data["round_time"] - exp_data["average_rtc_check_time"]
            )
            if limit is not None and fda_steps is not None:
                limit = fda_steps
                exp_data = exp_data.head(limit)
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm}")

        _ = get_column_max(exp_data, "distributed_acc")
        # Calculate cumulative adjusted round time in seconds and convert to minutes
        exp_data["cumulative_round_time"] = exp_data["total_round_time"].cumsum() / 60
        combined_data = pd.concat([combined_data, exp_data])
    # comment = f"ANN: {exp_data[group_by].values[0]} Batches processed: {fda_steps}"

    plotter.plot_combined_data(
        plot_type="line",
        x="cumulative_round_time",
        y="distributed_acc",
        data=combined_data,
        group_by=group_by,
        title="Distributed Accuracy vs Training Time",
        xlabel="Training Time (minutes)",
        ylabel="Distributed Accuracy",
        comment=comment,
    )


def plot_computation_communication_breakdown(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    group_by: str,
    comment: str = "",
    limit: int = None,
    width: float = 0.7,  # Adjust the width of the bars
):
    combined_data = pd.DataFrame()
    filename = "server.csv"
    dir = "server/"

    fda_steps = None
    for exp_id in experiment_ids:
        experiment_directory = os.path.join(root_directory, exp_id)
        file_directory = os.path.join(experiment_directory, dir)
        logging.info(f"Loading data for experiment ID: {exp_id}")
        data_handler.load_data(
            filename=filename,
            level="server",
            directory=file_directory,
        )

        exp_data = data_handler.get_data("server", clear_after=True)
        info_df = get_info_data(experiment_directory)
        exp_data[group_by] = info_df[group_by].values[0]
        algorithm = info_df["algorithm"].values[0]

        # Adjust communication time for synchronous algorithm
        if algorithm == "synchronous":
            exp_data["average_communication_time"] -= exp_data["average_rtc_check_time"]

        exp_data["algorithm"] = algorithm

        if limit is not None:
            if algorithm == "fda":
                limit += 1
                exp_data = exp_data.head(limit)
                fda_steps = get_total_steps(ROOT_DIRECTORY, exp_id, limit)
            elif algorithm == "synchronous" and fda_steps is not None:
                limit = fda_steps
                exp_data = exp_data.head(limit)

        _ = get_column_sum(exp_data, "average_communication_time")
        _ = get_column_sum(exp_data, "average_computation_time")
        _ = get_column_average(exp_data, "average_communication_time")
        _ = get_column_average(exp_data, "average_downlink_time")
        _ = get_column_average(exp_data, "average_uplink_time")
        _ = get_column_average(exp_data, "round_time")
        if algorithm == "fda":
            _ = get_column_average(exp_data, "average_rtc_check_time")

        print_network_avg_results_table(
            data_handler, plotter, experiment_ids, root_directory
        )
        combined_data = pd.concat([combined_data, exp_data])

    # Convert time from seconds to minutes
    combined_data["average_computation_time"] = (
        combined_data["average_computation_time"] / 60
    )
    combined_data["average_communication_time"] = (
        combined_data["average_communication_time"] / 60
    )

    # Aggregate data by summing the average computation and communication times
    aggregated_data = combined_data.groupby(group_by).sum().reset_index()

    x_label_opt = {
        "num_clients": "Number of clients",
        "threshold": "Threshold",
        "algorithm": "Algorithm",
        "client_mobility": "ClientMobility",
    }
    x_label = x_label_opt.get(group_by)

    plotter.plot_stacked_bar(
        data=aggregated_data,
        x=group_by,
        y=["average_computation_time", "average_communication_time"],
        title="Computation vs Communication Time Breakdown",
        xlabel=x_label,
        ylabel="Total Time (minutes)",
        labels=["Total Computation Time", "Total Communication Time"],
        comment=comment,
        width=width,
    )


def plot_steps_per_round(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    group_by: str,
    comment: str = "",
    limit: int = None,
):
    combined_data = pd.DataFrame()
    filename = "epoch.csv"
    dir = "clients"

    for exp_id in experiment_ids:
        experiment_directory = os.path.join(root_directory, exp_id, dir)
        client_dirs = [
            d
            for d in os.listdir(experiment_directory)
            if os.path.isdir(os.path.join(experiment_directory, d))
        ]

        for client_id in client_dirs:
            file_directory = os.path.join(experiment_directory, client_id)
            logging.info(
                f"Loading data for experiment ID: {exp_id}, Client ID: {client_id}"
            )
            data_handler.load_data(
                filename=filename,
                level="epoch",
                directory=file_directory,
                client_id=client_id,
            )

            exp_data = data_handler.get_data(
                "epoch", client_id=client_id, clear_after=True
            )

            info_df = get_info_data(os.path.join(root_directory, exp_id))
            exp_data[group_by] = info_df[group_by].values[0]

            if limit is not None:
                exp_data = exp_data.head(limit + 1)

            combined_data = pd.concat([combined_data, exp_data])

    plotter.plot_combined_data(
        plot_type="line",
        x="round",
        y="epoch_steps",
        data=combined_data,
        group_by=group_by,
        title="Steps Per Communication Round",
        xlabel="Round",
        ylabel="Epoch Steps",
        comment=comment,
    )

    plotter.plot_area(
        data=combined_data,
        x="round",
        y="epoch_steps",
        title="Epoch Steps per Epoch",
        xlabel="Round",
        ylabel="Epoch Steps",
        group_by=group_by,
        comment=comment,
    )


def plot_combined_accuracy(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    comment: str = "",
    limit: int = None,
):
    server_filename = "server.csv"
    server_dir = "server"

    for exp_id in experiment_ids:
        # Load server data
        server_directory = os.path.join(root_directory, exp_id, server_dir)
        data_handler.load_data(
            filename=server_filename,
            level="server",
            directory=server_directory,
        )

        server_data = data_handler.get_data("server", clear_after=True)

        # Get validation accuracy from the first client
        val_acc_data = get_val_acc(root_directory, exp_id, first=True)

        if limit is not None:
            server_data = server_data.head(limit + 1)
            val_acc_data = val_acc_data.head(limit + 1)

        # Combine server data with validation accuracy data
        combined_exp_data = pd.merge(
            server_data, val_acc_data, on=["round"], how="outer"
        )

    plotter.plot_line(
        data=combined_exp_data,
        x="round",
        y=["val_accuracy", "centralized_acc", "distributed_acc"],
        title="Validation, Centralized, and Distributed Accuracy over Rounds",
        xlabel="Round",
        ylabel="Accuracy",
        labels=["Validation Accuracy", "Centralized Accuracy", "Distributed Accuracy"],
    )


def plot_communication_time_contribution(df, plotter: DataPlotter):
    # Group by Network Template and Mobility
    df_grouped = (
        df.groupby(["Network Template", "Mobility", "Algorithm"]).mean().reset_index()
    )
    df_grouped["Template and Mobility"] = (
        df_grouped["Network Template"] + " - " + df_grouped["Mobility"].astype(str)
    )

    # Determine the columns to include in the plot based on the algorithm type
    y_columns = ["Average Downlink Time", "Average Uplink Time"]
    if "fda" in df_grouped["Algorithm"].values:
        y_columns.append("Average RTC Check Time")

    plotter.plot_stacked_bar(
        data=df_grouped,
        x="Template and Mobility",
        y=y_columns,
        title="Contribution of Each Component to Total Communication Time",
        xlabel="Network Template and Mobility",
        ylabel="Time (seconds)",
        labels=y_columns,
    )


def print_network_avg_results_table(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
):
    throughput_data = []

    for exp_id in experiment_ids:
        # Load info data
        experiment_directory = os.path.join(root_directory, exp_id)
        info_df = get_info_data(experiment_directory)
        algorithm = info_df["algorithm"].values[0]

        # Load server data
        server_directory = os.path.join(experiment_directory, "server")
        data_handler.load_data(
            filename="server.csv",
            level="server",
            directory=server_directory,
        )

        server_data = data_handler.get_data("server", clear_after=True)
        avg_throughput = server_data["average_throughput"].mean()
        avg_dropout_rate = server_data["dropouts"].mean()

        avg_downlink_time = server_data["average_downlink_time"].mean()
        avg_rtc_check_time = (
            server_data["average_rtc_check_time"].mean() if algorithm == "fda" else 0
        )
        avg_uplink_time = server_data["average_uplink_time"].mean()
        total_comm_time = avg_downlink_time + avg_rtc_check_time + avg_uplink_time

        template_opt = {0: "Weak Wifi", 1: "Medium Wifi", 2: "Fast Wifi"}
        template = template_opt.get(info_df["network_template"].values[0])

        throughput_data.append(
            {
                "Network Template": template,
                "Clients": info_df["num_clients"].values[0],
                "Mobility": info_df["client_mobility"].values[0],
                "Algorithm": algorithm,
                "Average Throughput": avg_throughput,
                "Average Dropout Rate": avg_dropout_rate,
                "Average Downlink Time": avg_downlink_time,
                "Average RTC Check Time": avg_rtc_check_time,
                "Average Uplink Time": avg_uplink_time,
                "Total Communication Time": total_comm_time,
            }
        )

    net_res_df = pd.DataFrame(throughput_data)
    print(net_res_df)

    plot_communication_time_contribution(net_res_df, plotter)


def plot_times_pie(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    comment: str = "",
):
    pie_data = prepare_pie_chart_data(data_handler, experiment_ids[0], root_directory)
    print(pie_data)
    plotter.plot_pie(
        data=pie_data,
        y="Time",
        title="Proportion of Communication Time Components",
        comment=comment,
    )


def varying_threshold(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    group_by: str,
    comment: str = "",
    limit: int = None,
):
    plot_accuracy_vs_round(
        data_handler, plotter, experiment_ids, root_directory, group_by, comment, limit
    )

    plot_accuracy_vs_training_time(
        data_handler, plotter, experiment_ids, root_directory, group_by, comment, limit
    )

    plot_computation_communication_breakdown(
        data_handler,
        plotter,
        experiment_ids,
        root_directory,
        group_by,
        comment,
        limit,
        width=10,
    )

    plot_steps_per_round(
        data_handler, plotter, experiment_ids, root_directory, group_by, comment, limit
    )


def fda_vs_sync(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    group_by: str,
    comment: str = "",
    limit: int = None,
):
    # plot_accuracy_vs_round(
    #     data_handler, plotter, experiment_ids, root_directory, group_by, comment, limit
    # )

    plot_accuracy_vs_training_time(
        data_handler, plotter, experiment_ids, root_directory, group_by, comment, limit
    )

    comment = ""
    plot_computation_communication_breakdown(
        data_handler, plotter, experiment_ids, root_directory, group_by, comment, limit
    )

    for exp_id in experiment_ids:
        exp_list = []
        exp_list.append(exp_id)
        plot_times_pie(data_handler, plotter, exp_list, root_directory, comment)


def non_iid(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    group_by: str,
    comment: str = "",
    limit: int = None,
):
    # We make significant assumptions as I know how I enter the arguments
    for i, exp_id in enumerate(experiment_ids):
        if i < 3:
            exp_ids = [exp_id for exp_id in [experiment_ids[i], experiment_ids[i + 3]]]
            # plot_accuracy_vs_training_time(
            #     data_handler, plotter, exp_ids, root_directory, group_by, comment, limit
            # )

            plot_accuracy_vs_round(
                data_handler,
                plotter,
                exp_ids,
                root_directory,
                group_by,
                comment,
                limit,
                centralized=True,
            )

    for i, exp_id in enumerate(experiment_ids):
        exp_list = [exp_id]
        plot_combined_accuracy(
            data_handler, plotter, exp_list, root_directory, comment, limit
        )


def anns_datasets(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    group_by: str,
    comment: str = "",
    limit: int = None,
):
    plot_accuracy_vs_round(
        data_handler, plotter, experiment_ids, root_directory, group_by, comment, limit
    )

    plot_accuracy_vs_training_time(
        data_handler, plotter, experiment_ids, root_directory, group_by, comment, limit
    )


def network_testing(
    data_handler: DataHandler,
    plotter: DataPlotter,
    experiment_ids: list,
    root_directory: str,
    group_by: str,
    comment: str = "",
    limit: int = None,
):
    # We make significant assumptions as I know how I enter the arguments
    for i in range(3):
        exp_ids = [
            exp_id for exp_id in [experiment_ids[2 * i], experiment_ids[2 * i + 1]]
        ]

        plot_accuracy_vs_training_time(
            data_handler,
            plotter,
            exp_ids,
            root_directory,
            group_by,
            comment,
            limit,
        )
        print_network_avg_results_table(data_handler, plotter, exp_ids, ROOT_DIRECTORY)
        plot_computation_communication_breakdown(
            data_handler, plotter, exp_ids, ROOT_DIRECTORY, group_by
        )

    print_network_avg_results_table(
        data_handler, plotter, experiment_ids, ROOT_DIRECTORY
    )
    return


def main():
    parser = argparse.ArgumentParser(
        description="Plot data for Federated Learning experiments"
    )
    parser.add_argument(
        "--experiment_ids", nargs="+", help="Experiment IDs to process", required=True
    )
    parser.add_argument(
        "--group_by",
        help="What to use as grouping factor",
        required=True,
        choices=[
            "threshold",
            "algorithm",
            "num_clients",
            "client_mobility",
            "ann",
            "network_template",
        ],
    )

    parser.add_argument(
        "--mode",
        help="Experiment Setup",
        required=True,
        choices=["1", "2", "3", "4", "5"],
    )

    parser.add_argument(
        "--limit",
        help="Number of rounds to depict",
        required=False,
    )
    args = parser.parse_args()

    group_by = args.group_by
    limit = int(args.limit) if args.limit is not None else None

    data_handler = DataHandler()
    plotter = DataPlotter()

    experiment_ids = args.experiment_ids

    comment = ""

    # Dictionary to map mode numbers to functions
    mode_functions = {
        "1": lambda: fda_vs_sync(
            data_handler,
            plotter,
            experiment_ids,
            ROOT_DIRECTORY,
            group_by,
            comment,
            limit,
        ),
        "2": lambda: varying_threshold(
            data_handler, plotter, experiment_ids, ROOT_DIRECTORY, group_by, comment
        ),
        "3": lambda: anns_datasets(
            data_handler,
            plotter,
            experiment_ids,
            ROOT_DIRECTORY,
            group_by=group_by,
            limit=limit,
            comment=comment,
        ),
        "4": lambda: non_iid(
            data_handler,
            plotter,
            experiment_ids,
            ROOT_DIRECTORY,
            group_by=group_by,
            comment=comment,
        ),
        "5": lambda: network_testing(
            data_handler,
            plotter,
            experiment_ids,
            ROOT_DIRECTORY,
            group_by=group_by,
            comment=comment,
        ),
    }

    # Execute the function corresponding to the selected mode
    selected_mode = args.mode
    mode_functions[selected_mode]()


if __name__ == "__main__":
    main()
