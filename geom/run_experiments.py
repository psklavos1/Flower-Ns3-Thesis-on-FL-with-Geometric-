import subprocess
import time
import os
import yaml
import logging

PARAM_FILE = "experiment_params.yaml"
SERVER_LOG_FILE = "/tmp/server_screen.log"  # Define a log file path
N3_PORT = 9090  # The NS3 port

SERVER_INIT_INTERVAL = 20  # Adjust this delay based on server initialization time
BETWEEN_CLIENT_INTERVAL = 0.25  # Adjust this delay
BETWEEN_EXPERIMENTS_INTERVAL = 10
CHECK_COMPLETION_INTERVAL = 20

SERVER_SESSION_NAME = "Server"
CLIENT_SESSION_NAME = "Clients"
NS3_SESSION_NAME = "NS3"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def free_port(port):
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"], capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                pid = int(parts[1])
                subprocess.run(["kill", "-9", str(pid)])
                logging.info(f"Process {pid} using port {port} has been terminated.")
    except Exception as e:
        logging.error(f"Failed to free port {port}: {e}")


def start_screen_session(name, command, log_file):
    try:
        subprocess.run(
            [
                "screen",
                "-L",
                "-Logfile",
                log_file,
                "-dmS",
                name,
                "bash",
                "-c",
                f"{command}; exec bash",
            ]
        )
        logging.info(f"Started screen session {name} with command: {command}")
    except Exception as e:
        logging.error(f"Failed to start screen session {name}: {e}")


def kill_screen_session(session_name):
    subprocess.run(
        ["screen", "-S", session_name, "-X", "quit"], stderr=subprocess.DEVNULL
    )
    logging.info(f"Terminated screen session {session_name}.")


def kill_all_screen_sessions():
    kill_screen_session(SERVER_SESSION_NAME)
    kill_screen_session(CLIENT_SESSION_NAME)
    kill_screen_session(NS3_SESSION_NAME)


def cleanup():
    kill_all_screen_sessions()
    remove_log_file(SERVER_LOG_FILE)
    free_port(N3_PORT)


def run_experiment(params, id):
    num_clients = params["num_clients"]
    if id == 0:
        cleanup()

    server_command = (
        f"source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate geom_aio; "
        f"python server_main.py "
        f"num_clients={params['num_clients']} "
        f"dataset={params['dataset']} "
        f"ann={params['ann']} "
        f"threshold={params['threshold']} "
        f"thres_discount_factor={params['thres_discount_factor']} "
        f"steps_threshold={params['steps_threshold']} "
        f"flower_server_cfg.num_rounds={params['rounds']} "
        f"flower_server_cfg.fit_cfg.batch_size={params['train_batch_size']} "
        f"flower_server_cfg.eval_cfg.batch_size={params['eval_batch_size']} "
        f"ns3.wifi_net_template={params['network_template']} "
        f"ns3.moving_clients={params['client_mobility']} "
        f"non_iid={params['non_iid']} "
        f"bias_template={params['bias_template']}"
    )

    start_screen_session(SERVER_SESSION_NAME, server_command, SERVER_LOG_FILE)
    logging.info(f"Started server in screen session: {SERVER_SESSION_NAME}")

    time.sleep(SERVER_INIT_INTERVAL)

    subprocess.run(["screen", "-dmS", CLIENT_SESSION_NAME])
    logging.info(f"Started main client screen session: {CLIENT_SESSION_NAME}")

    for i in range(num_clients):
        client_window_name = f"Client_{i+1}"
        client_command = (
            f"source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate geom_aio; "
            f"python client_main.py "
            f"partition_id={i} "
            f"num_clients={params['num_clients']} "
            f"dataset={params['dataset']} "
            f"ann={params['ann']} "
            f"non_iid={params['non_iid']} "
            f"bias_template={params['bias_template']} "
        )
        if not params["non_iid"]:
            client_command += f"keep_log={(i == 0)}"
        else:
            client_command += f"keep_log={(i == 0 or i ==1)}"

        subprocess.run(
            [
                "screen",
                "-S",
                CLIENT_SESSION_NAME,
                "-X",
                "screen",
                "-t",
                client_window_name,
                "bash",
                "-c",
                f"{client_command}; exec bash",
            ]
        )
        logging.info(f"Started client {i+1} in screen window: {client_window_name}")
        time.sleep(BETWEEN_CLIENT_INTERVAL)


def remove_log_file(log_file):
    try:
        if os.path.exists(log_file):
            os.remove(log_file)
            logging.info(f"Removed log file {log_file}")
    except Exception as e:
        logging.error(f"Error removing log file {log_file}: {e}")


def wait_for_experiment_completion(log_file):
    retval = -1
    while True:
        with open(log_file, "r") as f:
            log_content = f.read()
            if "Experiment Over" in log_content:
                retval = 0
                break
        logging.info("Waiting for server to complete...")
        time.sleep(BETWEEN_EXPERIMENTS_INTERVAL)

    logging.info("Server session has terminated. Experiment completed.")
    return retval


def load_params(filename: str) -> dict:
    try:
        with open(filename, "r") as file:
            experiments_params = yaml.safe_load(file)
            if experiments_params is None:
                raise ValueError("The parameter file is empty or malformed.")
            return experiments_params
    except FileNotFoundError:
        logging.error(f"Error: The file '{filename}' was not found.")
        return None
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing the parameter file: {exc}")
        return None
    except Exception as exc:
        logging.error(f"An error occurred: {exc}")
        return None


def main():
    experiment_params = load_params(PARAM_FILE)
    if not experiment_params:
        return

    for experiment_no, params in enumerate(experiment_params):
        if params.get("completed"):
            logging.info(f"Skipping completed experiment {experiment_no + 1}")
            continue

        logging.info(f"Starting experiment {experiment_no + 1}")
        run_experiment(params, experiment_no)
        retval = wait_for_experiment_completion(SERVER_LOG_FILE)
        cleanup()

        if retval == 0:
            params["completed"] = True

        with open(PARAM_FILE, "w") as file:
            yaml.dump(experiment_params, file)

        time.sleep(BETWEEN_EXPERIMENTS_INTERVAL)


if __name__ == "__main__":
    main()
