import yaml
import subprocess
import time
import os

PARAMS_FILE = "experiment_params.yaml"
SERVER_LOG_FILE = "/tmp/server_screen.log"  # Define a log file path
N3_PORT = 9090  # THe NS3 port

SERVER_INIT_INTERVAL = 10  # Adjust this delay based on server initialization time
BETWEEN_CLIENT_INTERVAL = 0.25  # Adjust this delay
BETWEEN_EXPERIMENTS_INTERVAL = 10
CHECK_COMPLETION_INTERVAL = 15


SERVER_SESSION_NAME = "Server"
CLIENT_SESSION_NAME = "Clients"
NS3_SESSION_NAME = "NS3"


def free_port(port):
    # Find the process using the port
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}"], capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:  # The first line is the header
            for line in lines[1:]:
                parts = line.split()
                pid = int(parts[1])
                # Kill the process
                subprocess.run(["kill", "-9", str(pid)])
                print(f"Process {pid} using port {port} has been terminated.")
    except Exception as e:
        print(f"Failed to free port {port}: {e}")


def generate_experiment_params():
    experiments_params = []
    for num_clients in [5]:
        for dataset in ["mnist"]:
            for ann in ["lenet"]:
                params = {
                    "num_clients": num_clients,
                    "dataset": dataset,
                    "ann": ann,
                    "threshold": 5,
                    "thres_discount_factor": 1.0,
                    "steps_threshold": 500,
                    "train_batch_size": 32,
                    "eval_batch_size": 64,
                    "network_template": 2,
                    "client_mobility": False,
                    "non_iid": False,
                    "bias_template": 0,
                    "completed": False,  # Field to track if the experiment is completed
                }
                experiments_params.append(params)
    return experiments_params


def save_experiment_params(params, filename=PARAMS_FILE):
    with open(filename, "w") as file:
        yaml.dump(params, file)


def load_experiment_params(filename=PARAMS_FILE):
    if os.path.exists(filename):
        with open(filename, "r") as file:
            return yaml.safe_load(file)
    return []


def start_screen_session(name, command, log_file):
    # Start a new detached screen session with the given name and command, with logging enabled
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


def kill_screen_session(session_name):
    subprocess.run(
        ["screen", "-S", session_name, "-X", "quit"], stderr=subprocess.DEVNULL
    )

    print(f"Terminated screen session {session_name}.")


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

    # Start the server in a separate screen session and log output
    server_command = (
        f"source /home/psklavos/miniconda3/etc/profile.d/conda.sh; conda activate geom_aio; "
        f"python server_main.py "
        f"num_clients={params['num_clients']} "
        f"dataset={params['dataset']} "
        f"ann={params['ann']} "
        f"threshold={params['threshold']} "
        f"thres_discount_factor={params['thres_discount_factor']} "
        f"steps_threshold={params['steps_threshold']} "
        f"flower_server_cfg.fit_cfg.batch_size={params['train_batch_size']} "
        f"flower_server_cfg.eval_cfg.batch_size={params['eval_batch_size']} "
        f"ns3.wifi_net_template={params['network_template']} "
        f"ns3.moving_clients={params['client_mobility']} "
        f"non_iid={params['non_iid']} "
        f"bias_template={params['bias_template']}"
    )

    start_screen_session(SERVER_SESSION_NAME, server_command, SERVER_LOG_FILE)
    print(f"Started server in screen session: {SERVER_SESSION_NAME}")

    time.sleep(SERVER_INIT_INTERVAL)

    subprocess.run(["screen", "-dmS", CLIENT_SESSION_NAME])
    print(f"Started main client screen session: {CLIENT_SESSION_NAME}")

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
        print(f"Started client {i+1} in screen window: {client_window_name}")
        time.sleep(BETWEEN_CLIENT_INTERVAL)


def remove_log_file(log_file):
    try:
        if os.path.exists(log_file):
            os.remove(log_file)
    except Exception as e:
        print(f"Error removing log file {log_file}: {e}")


def wait_for_experiment_completion(log_file):
    retval = -1

    while True:
        # Read the log file to check for the shutdown message
        with open(log_file, "r") as f:
            log_content = f.read()
            if (
                "Experiment Over" in log_content
            ):  # Check for the specific shutdown message
                retval = 0
                break

        print("Waiting for server to complete...")
        time.sleep(BETWEEN_EXPERIMENTS_INTERVAL)

    print("Server session has terminated. Experiment completed.")
    return retval


def main():
    experiments_params = load_experiment_params()

    if not experiments_params:
        experiments_params = generate_experiment_params()
        save_experiment_params(experiments_params)

    for experiment_no, params in enumerate(experiments_params):
        if params.get("completed"):
            print(f"Skipping completed experiment {experiment_no + 1}")
            continue

        print(f"Starting experiment {experiment_no + 1}")
        run_experiment(params, experiment_no)
        wait_for_experiment_completion(SERVER_LOG_FILE)
        cleanup()

        params["completed"] = True
        save_experiment_params(experiments_params)

        time.sleep(BETWEEN_EXPERIMENTS_INTERVAL)


if __name__ == "__main__":
    main()
