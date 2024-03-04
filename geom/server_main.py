# third party
import hydra
from omegaconf import DictConfig, OmegaConf

# built-in
import threading

# local
from server.flower_server import FlowerServer
from server.metric_server import MetricServer
from utils.monitor import Monitor


@hydra.main(config_path="conf", config_name="server", version_base="1.1")
def server_main(cfg: DictConfig):
    print("\n\n=================== Server Config File ===================\n")
    print(OmegaConf.to_yaml(cfg))
    print("=================== End ===================\n")

    monitor = Monitor()
    # Thread for Flower server
    # print(OmegaConf.to_yaml(cfg.network))

    flower_server = FlowerServer(cfg.flower_server_cfg, cfg.ns3, monitor)
    flower_thread = threading.Thread(target=flower_server.start)
    flower_thread.start()
    print("Flower Server Starts")

    # Thread for gRPC server
    metric_server = MetricServer(cfg.metric_server_cfg, monitor)
    metric_thread = threading.Thread(target=metric_server.start)
    metric_thread.start()
    print("Metric Server Starts")

    flower_thread.join()
    metric_thread.join()


if __name__ == "__main__":
    server_main()
