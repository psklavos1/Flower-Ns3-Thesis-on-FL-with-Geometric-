# third party
import flwr as fl
import hydra
from omegaconf import DictConfig, OmegaConf

# local
from client.client import FlexibleClient


@hydra.main(config_path="conf", config_name="client")
def client_main(cfg: DictConfig):
    print("\n\n=================== Config File ===================\n")
    print(OmegaConf.to_yaml(cfg))
    print("===========================================\n")

    fl.client.start_client(
        server_address=cfg.server_address,
        client=FlexibleClient(
            num_clients=cfg.num_clients,
            partition_id=cfg.partition_id,
            ds_name=cfg.dataset,
            non_iid=cfg.non_iid,
            bias_template=cfg.bias_template,
        ).to_client(),  # <-- where FlowerClient is of type flwr.client.NumPyClient object
    )


if __name__ == "__main__":
    client_main()
