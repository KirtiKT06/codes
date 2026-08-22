from ml.datasets.mnist_loader import load_data_wrapper
from ml.models.network8 import network
from ml.utils.experiment_logger import ExperimentLogger
from ml.utils.config_loader import load_config


def main():

    # --- load config first ---
    cfg = load_config("mnist_baseline.yaml")
    tcfg = cfg["training"]

    # --- logger ---
    logger = ExperimentLogger(
        exp_name=cfg["experiment"]["name"],
        config_path="ml/configs/mnist_baseline.yaml"
    )

    # --- data ---
    training_data, validation_data, test_data = load_data_wrapper()

    # --- model ---
    net = network(cfg["model"]["layers"])

    # --- train ---
    net.SGD(
        training_data=training_data,
        epochs=tcfg["epochs"],
        mini_batch_size=tcfg["batch_size"],
        eta=tcfg["eta"],
        lmbda=tcfg["lmbda"],
        gamma=tcfg["gamma"],
        test_data=test_data,
        monitor_evaluation_accuracy=True,
        logger=logger
    )

    # --- log basic metrics ---
    logger.log_metric("epochs", tcfg["epochs"])
    logger.log_metric("batch_size", tcfg["batch_size"])
    logger.log_metric("eta", tcfg["eta"])


if __name__ == "__main__":
    main()
