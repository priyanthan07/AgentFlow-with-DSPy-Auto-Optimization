import asyncio
import sys
from pathlib import Path
import subprocess
import threading

from src.experiments.experiment_runner import ExperimentRunner
from src.utils.logger import get_logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

logger = get_logger(__name__)

def start_mlflow_ui():
    def run_ui():
        try:
            subprocess.Popen(["mlflow", "ui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            raise e
    threading.Thread(target=run_ui, daemon=True).start()

async def main():
    
    start_mlflow_ui()
    
    training_size = 5
    test_size = 3

    # Initialize experiment runner
    runner = ExperimentRunner()
    logger.info(" Check MLflow UI: http://localhost:5000")
    try:
        results = await runner.run_full_experiment(
            training_size = training_size,
            test_size = test_size
        )

        logger.info(" Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
