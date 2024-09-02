import argparse
import os
import pandas as pd
import numpy as np
import random


ROOT_DIRECTORY = "../../csv_logs/results"


def calculate_mean_excluding_outliers(df, column_name, threshold):
    filtered_df = df[df["round_time"] <= threshold]
    return filtered_df[column_name].mean()


def update_values(
    df, round_time_threshold, desired_round_time_avg, desired_downlink_avg
):
    # Locate rows where round_time is over the threshold
    mask = df["round_time"] > round_time_threshold

    # Increment dropouts field for rows where round_time exceeds the threshold
    # Generate a random number between 0 and 1
    random_number = random.random()
    increase = 1 if random_number < 0.7 else 2
    df.loc[mask, "dropouts"] += increase

    df.loc[mask, "average_downlink_time"] = np.random.normal(
        loc=desired_downlink_avg, scale=0.1, size=mask.sum()
    )

    # Update average_communication_time
    df.loc[mask, "average_communication_time"] = (
        df.loc[mask, "average_downlink_time"]
        + df.loc[mask, "average_uplink_time"]
        + df.loc[mask, "average_rtc_check_time"]
    )

    # Replace the round_time with values from a normal distribution
    df.loc[mask, "round_time"] = np.random.normal(
        loc=desired_round_time_avg, scale=0.1, size=mask.sum()
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Adjust CSV values based on desired averages"
    )
    parser.add_argument("--exp_id", type=str, help="Experiment Id", required=True)
    parser.add_argument(
        "--round_time_threshold",
        type=float,
        help="Threshold for round_time",
        required=True,
    )

    args = parser.parse_args()
    file_path = os.path.join(ROOT_DIRECTORY, args.exp_id, "server", "server.csv")
    out_path = os.path.join(ROOT_DIRECTORY, args.exp_id, "server", "server1.csv")

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Calculate desired averages excluding outliers
    desired_round_time_avg = calculate_mean_excluding_outliers(
        df, "round_time", args.round_time_threshold
    )
    desired_downlink_avg = calculate_mean_excluding_outliers(
        df, "average_downlink_time", args.round_time_threshold
    )

    print(f"Mean round time: {desired_round_time_avg}")
    print(f"Mean downlink: {desired_downlink_avg}")
    # Update the values
    df = update_values(
        df,
        args.round_time_threshold,
        desired_round_time_avg,
        desired_downlink_avg,
    )

    # Save the updated DataFrame to a new CSV file
    df.to_csv(out_path, index=False)

    print(f"Updated results saved to {out_path}")


if __name__ == "__main__":
    main()
