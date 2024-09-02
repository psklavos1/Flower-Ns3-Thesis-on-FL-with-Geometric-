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
    print("====================== End ========================\n")
    # Define gRPC channel options for max message length

    client = FlexibleClient(
        num_clients=cfg.num_clients,
        partition_id=cfg.partition_id,
        ds_name=cfg.dataset,
        ann_name=cfg.ann,
        non_iid=cfg.non_iid,
        bias_template=cfg.bias_template,
        val_ratio=cfg.val_ratio,
        train_val_ratio=cfg.train_val_ratio,
        keep_log=cfg.keep_log,
    )
    fl.client.start_client(
        server_address=cfg.server_address,
        client=client.to_client(),
        grpc_max_message_length=1024 * 1024 * 1024,  # <-- where FlowerClient is of type flwr.client.NumPyaClient object
    )


if __name__ == "__main__":
    client_main()
