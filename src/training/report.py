import json
import time

from src.utils.paths import DOCS_DIR


class TrainingReport:
    """
    Generates a training experiment summary report.
    """

    def generate(
        self,
        corpus_stats,
        model_config,
        training_config,
        losses,
        samples,
        wall_time_s=None,
    ):

        wall_time_s = wall_time_s or round(
            time.time(),
            3,
        )

        stride = max(len(losses) // 100, 1)

        report = {
            "corpus": corpus_stats,
            "model": model_config,
            "training": training_config,
            "initial_loss": round(losses[0], 4),
            "final_loss": round(losses[-1], 4),
            "min_loss": round(min(losses), 4),
            "loss_curve": [
                round(loss, 4)
                for loss in losses[::stride]
            ],
            "samples": samples,
            "wall_time_s": wall_time_s,
        }

        report_path = (
            DOCS_DIR
            / "reports"
            / "training_report.json"
        )

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
            )

        return report
