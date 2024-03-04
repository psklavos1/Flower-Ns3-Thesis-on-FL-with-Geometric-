import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="common", version_base="1.1")
def get_num_clients(cfg: DictConfig):
    print(cfg.num_clients)


if __name__ == "__main__":
    get_num_clients()
