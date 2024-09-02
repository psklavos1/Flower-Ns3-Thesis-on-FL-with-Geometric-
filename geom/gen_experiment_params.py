# experiment_params.py
import yaml


# * DONE * #
# * Test Threshold values -> 4 expers
def threshold_expers():
    experiments_params = []
    for threshold in [80, 100, 120, 140]:
        params = {
            "num_clients": 12,
            "dataset": "fashion_mnist",
            "ann": "",  # !
            "threshold": threshold,  # !
            "thres_discount_factor": 0.99,
            "steps_threshold": 10000,
            "train_batch_size": 64,
            "eval_batch_size": 128,
            "network_template": 1,
            "client_mobility": False,
            "non_iid": False,
            "rounds": 40,
            "bias_template": 0,
            "completed": False,  # Field to track if the experiment is completed
        }
        experiments_params.append(params)
    return experiments_params


# * DONE * #
# * Test num_Clients -> 4 expers
def num_clients_expers():
    experiments_params = []
    for num_clients in [5, 10, 15, 20]:
        params = {
            "num_clients": num_clients,
            "dataset": "fashion_mnist",
            "ann": "vgg",  # !
            "threshold": 50,  # !
            "thres_discount_factor": 0.9875,
            "steps_threshold": 10000,
            "train_batch_size": 64,
            "eval_batch_size": 128,
            "network_template": 1,
            "client_mobility": False,
            "non_iid": True,
            "rounds": 50,
            "bias_template": 0,
            "completed": False,  # Field to track if the experiment is completed
        }
        experiments_params.append(params)
    return experiments_params


# * DONE * #
# * Test Non-iid different amount of clients-> 6
def non_iid_expers():
    experiments_params = []
    for num_clients in [5, 15]:
        for bias_template in [1, 2, 3]:
            params = {
                "num_clients": num_clients,
                "dataset": "fashion_mnist",
                "ann": "",
                "threshold": 80,
                "thres_discount_factor": 0.99,
                "steps_threshold": 10000,
                "train_batch_size": 64,
                "eval_batch_size": 128,
                "network_template": 2,
                "client_mobility": False,
                "non_iid": True,
                "rounds": 40,
                "bias_template": bias_template,
                "completed": False,  # Field to track if the experiment is completed
            }
            experiments_params.append(params)
    return experiments_params


# * 2 Datasets * 2 ANNs -> 4 expers
def dataset_ann_expers():
    experiments_params = []
    # Cifar 10
    for dataset, ann, threshold in [
        ("cifar10", "vgg", 1000),
        ("cifar10", "resnet", 800),
        # ("mnist", "lenet", 3),
        # ("mnist", "", 8),
    ]:  # ! SOS TO TEST
        params = {
            "num_clients": 10,
            "dataset": dataset,
            "ann": ann,
            "threshold": threshold,
            "thres_discount_factor": 0.99,
            "steps_threshold": 10000,
            "train_batch_size": 64,
            "eval_batch_size": 128,
            "network_template": 2,
            "client_mobility": False,
            "non_iid": False,
            "rounds": 50,
            "bias_template": 0,
            "completed": False,  # Field to track if the experiment is completed
        }
        experiments_params.append(params)
    return experiments_params


# * DONE * #
# *Test wifi_templates -> 6 expers
def network_simulator_expers():
    experiments_params = []
    for wifi_template in [0, 1, 2]:  # 3
        for moving in [False, True]:
            params = {
                "num_clients": 15,
                "dataset": "fashion_mnist",
                "ann": "",  # !
                "threshold": 80,  # !
                "thres_discount_factor": 0.99,
                "steps_threshold": 10000,
                "train_batch_size": 64,
                "eval_batch_size": 128,
                "network_template": wifi_template,
                "client_mobility": moving,
                "non_iid": False,
                "rounds": 20,
                "bias_template": 0,
                "completed": False,  # Field to track if the experiment is completed
            }
            experiments_params.append(params)
    return experiments_params


# ? SoS Experiment with CIFAR-100 -> 1
def cifar_100_exper():
    experiments_params = {
        "num_clients": 12,
        "dataset": "cifar100",
        "ann": "resnet",
        "threshold": 75,
        "thres_discount_factor": 0.9875,
        "steps_threshold": 10000,
        "train_batch_size": 64,
        "eval_batch_size": 128,
        "network_template": 2,
        "client_mobility": False,
        "non_iid": False,
        "rounds": 40,
        "bias_template": 0,
        "completed": False,  # Field to track if the experiment is completed
    }
    return experiments_params


# * Side Expers: 25


# * DONE * #
# * 4+4 = 8 expers
def base_expers(synchronous=False):
    threshold = 10
    if synchronous:
        threshold = 0
    experiments_params = []
    for num_clients in [5, 10, 15, 20]:  # 4
        params = {
            "num_clients": num_clients,
            "dataset": "fashion_mnist",
            "ann": "lenet",
            "threshold": threshold,
            "thres_discount_factor": 0.99,
            "steps_threshold": 10000,
            "train_batch_size": 64,
            "eval_batch_size": 128,
            "network_template": 1,
            "client_mobility": False,
            "non_iid": False,
            "rounds": 40,
            "bias_template": 0,
            "completed": False,  # Field to track if the experiment is completed
        }
        experiments_params.append(params)
    return experiments_params


def save_experiment_params(params, filename):
    with open(filename, "w") as file:
        yaml.dump(params, file)
    print(f"Experiment parameters saved to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment parameters.")
    parser.add_argument(
        "filename",
        type=str,
        help="The name of the file to save the parameters (!Without prefix).",
    )

    args = parser.parse_args()

    # params = base_expers()
    # params.extend(network_simulator_expers())
    # params.extend(threshold_expers())
    params = network_simulator_expers()

    filename = args.filename + ".yaml"
    save_experiment_params(params, args.filename)
