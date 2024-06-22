import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose

"""
This is a utility module used to extract the number of clients from the configuration files to the run.sh script
that is used to initiate an experiment
"""


@hydra.main(config_path="conf", config_name="common", version_base="1.1")
def get_num_clients(cfg: DictConfig):
    print(cfg.num_clients)

if __name__ == "__main__":
    # Ensure GlobalHydra is clear before initializing
    GlobalHydra.instance().clear()

    # Initialize Hydra with specific configurations to disable output directory creation
    with initialize(config_path="conf"):
        cfg = compose(config_name="common", overrides=["hydra.run.dir=.", "hydra.output_subdir=null"])
        print(cfg.num_clients)