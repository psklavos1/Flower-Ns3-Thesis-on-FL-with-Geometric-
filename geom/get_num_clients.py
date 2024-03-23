import hydra
from omegaconf import DictConfig

"""
This is a utility module used to extract the number of clients from the configuration files to the run.sh script
that is used to initiate an experiment
"""


@hydra.main(config_path="conf", config_name="common", version_base="1.1")
def get_num_clients(cfg: DictConfig):
    print(cfg.num_clients)


if __name__ == "__main__":
    get_num_clients()
