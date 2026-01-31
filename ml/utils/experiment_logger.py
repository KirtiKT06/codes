import json
import shutil
from datetime import datetime
from pathlib import Path


class ExperimentLogger:

    def __init__(self, exp_name: str, config_path: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("results") / f"{exp_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # save config snapshot
        shutil.copy(config_path, self.run_dir / "config.yaml")

        self.metrics_path = self.run_dir / "metrics.json"
        self.metrics = {}

    def log_metric(self, name: str, value):
        self.metrics[name] = value
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def log_text(self, filename: str, text: str):
        with open(self.run_dir / filename, "w") as f:
            f.write(text)
