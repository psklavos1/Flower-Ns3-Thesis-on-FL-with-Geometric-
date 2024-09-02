from data_handler import DataHandler


def main():
    root_dir = "../../csv_logs/results"  # Assuming the script is run from the root directory containing the exp_id directories
    output_file = "../../csv_logs/results/all_info.csv"
    data_handler = DataHandler()

    all_info_df = data_handler.aggregate_info_files(root_dir)
    all_info_df.to_csv(output_file, index=False)
    print(f"Aggregated information saved to {output_file}")


if __name__ == "__main__":
    main()
