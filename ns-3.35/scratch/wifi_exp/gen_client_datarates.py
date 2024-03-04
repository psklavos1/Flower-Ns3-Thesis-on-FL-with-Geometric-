import os
import random
import hydra
from omegaconf import DictConfig


def count_lines(filepath):
    """Count the number of lines in a file."""
    with open(filepath, "r") as file:
        lines = file.readlines()
        return len(lines)


def generate_client_datarate(min, max, input_filepath, output_filepath):
    """Generate a file with lines as many as the clients in input file, each containing a number between MIN and MAX."""
    num_lines = count_lines(input_filepath)
    with open(output_filepath, "w") as file:
        for _ in range(num_lines - 1):
            # Generate a random value
            # ? could be utilizing gaussian distribution maybe
            val = random.randint(min, max)
            file.write(f"{val}kbps\n")
    return num_lines - 1


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


@hydra.main(config_path="conf", config_name="datarate", version_base="1.1")
def main(cfg: DictConfig):

    # Print the current working directory
    min = cfg.min
    max = cfg.max
    input_filepath = cfg.input_filepath
    output_filepath = cfg.output_filepath  # The output file name

    num_lines = generate_client_datarate(min, max, input_filepath, output_filepath)
    print(f"Generated file with {num_lines} lines at '{output_filepath}'")

    # Put in Comment If needed
    log_file = "./gen_client_datarates.log"
    delete_file(log_file)


if __name__ == "__main__":
    main()
