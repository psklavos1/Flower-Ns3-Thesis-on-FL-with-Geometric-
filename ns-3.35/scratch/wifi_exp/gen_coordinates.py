import os
import random
import hydra
from omegaconf import DictConfig


def generate_coordinates(
    file_path, min_valueX, max_valueX, min_valueY, max_valueY, number_of_pairs
):
    """Generate a file with pairs of numbers in the specified range."""
    with open(file_path, "w") as file:
        for _ in range(number_of_pairs):
            # Generate two random numbers within the specified range
            number1 = random.uniform(min_valueX, max_valueX)
            number2 = random.uniform(min_valueY, max_valueY)

            # Write the numbers to the file, formatted as specified
            file.write(f"{number1:.2f}\t{number2:.2f}\n")


def delete_file(filename):
    """Delete a file with the specified filename."""
    try:
        os.remove(filename)
        # print(f"File '{filename}' has been deleted.")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except PermissionError:
        print(f"Permission denied: cannot delete file '{filename}'.")
    except Exception as e:
        print(f"Error deleting file '{filename}': {e}")


@hydra.main(config_path="conf", config_name="coordinates", version_base="1.1")
def main(cfg: DictConfig):

    # Print the current working directory
    num_nodes = cfg.num_nodes
    minX = cfg.minX
    maxX = cfg.maxX
    minY = cfg.minY
    maxY = cfg.maxY
    # The paths given are relative to the execution path. So run from cmd not vsCode
    output_filepath = cfg.output_path

    generate_coordinates(output_filepath, minX, maxX, minY, maxY, num_nodes)
    print(f"Generated file '{output_filepath}' with {num_nodes} coordinates")

    log_file = "./gen_coordinates.log"
    delete_file(log_file)


if __name__ == "__main__":
    main()
