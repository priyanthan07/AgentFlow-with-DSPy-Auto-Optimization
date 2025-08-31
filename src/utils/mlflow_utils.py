import mlflow
import mlflow.dspy
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from src.config import MLFLOW_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowManager:
    """Centralized MLflow experiment management."""

    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or MLFLOW_CONFIG["experiment_name"]
        self.tracking_uri = MLFLOW_CONFIG["tracking_uri"]
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            # Create experiment if it doesn't exist
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    mlflow.create_experiment(
                        name=self.experiment_name,
                        artifact_location=MLFLOW_CONFIG["artifact_location"],
                    )
                    logger.info(f"Created MLflow experiment: {self.experiment_name}")

                mlflow.set_experiment(self.experiment_name)

            except Exception as e:
                logger.error(f"Error setting up MLflow experiment: {e}")

        except Exception as e:
            logger.error(f"Error connecting to MLflow: {e}")

    def start_optimization_run(self, run_name: str, config: Dict[str, Any]):
        """Start an optimization run with configuration logging."""
        run = mlflow.start_run(
            run_name=f"Optimization_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Log configuration parameters
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))

        # Log system info
        mlflow.log_param("optimization_timestamp", datetime.now().isoformat())

        return run

    def start_evaluation_run(self, run_name: str, agent_type: str):
        """Start an evaluation run."""
        run = mlflow.start_run(
            run_name=f"Evaluation_{agent_type}_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        mlflow.log_param("agent_type", agent_type)
        mlflow.log_param("evaluation_timestamp", datetime.now().isoformat())

        return run

    def log_agent_performance(self, results: List[Dict[str, Any]], agent_type: str):
        """Log agent performance metrics."""
        if not results:
            return

        # Calculate aggregate metrics
        avg_quality = sum(r["quality_score"] for r in results) / len(results)
        avg_sources = sum(r["sources_analyzed"] for r in results) / len(results)
        avg_findings = sum(r["key_findings_count"] for r in results) / len(results)
        avg_iterations = sum(r["iterations"] for r in results) / len(results)

        # Log aggregate metrics
        mlflow.log_metric(f"{agent_type}_avg_quality_score", avg_quality)
        mlflow.log_metric(f"{agent_type}_avg_sources_analyzed", avg_sources)
        mlflow.log_metric(f"{agent_type}_avg_key_findings", avg_findings)
        mlflow.log_metric(f"{agent_type}_avg_iterations", avg_iterations)

        # Log individual results as artifact
        results_file = f"{agent_type}_detailed_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        mlflow.log_artifact(results_file)
        os.remove(results_file)  # Clean up temp file

    def log_comparison_results(self, improvements: Dict[str, float]):
        """Log agent comparison results."""
        for metric, improvement in improvements.items():
            mlflow.log_metric(f"improvement_{metric}_percent", improvement)

        # Log overall improvement score
        avg_improvement = sum(improvements.values()) / len(improvements)
        mlflow.log_metric("overall_improvement_percent", avg_improvement)

    def load_optimized_agent(
        self, agent_class, agent_name: str, version: str = "latest"
    ):
        """Load optimized agent from storage."""
        agents_dir = Path("src/optimized_agents")

        if version == "latest":
            agent_file = agents_dir / f"{agent_name}_latest.json"
        else:
            agent_file = agents_dir / f"{agent_name}_optimized_{version}.json"

        if not agent_file.exists():
            logger.error(f"Optimized agent not found: {agent_file}")
            return None

        try:
            agent = agent_class()
            agent.load(str(agent_file))
            logger.info(f"Loaded optimized agent: {agent_file}")
            return agent
        except Exception as e:
            logger.error(f"Error loading optimized agent: {e}")
            return None


class DSPyMLflowIntegration:
    """Integration utilities for DSPy and MLflow."""

    @staticmethod
    def setup_dspy_autolog():
        """Setup DSPy autologging with MLflow."""
        try:
            mlflow.dspy.autolog(
                log_compiles=True,
                log_evals=True,
                log_traces_from_compile=True,
                silent=False,
            )
            logger.info("DSPy autologging enabled")
        except Exception as e:
            logger.error(f"Error setting up DSPy autologging: {e}")

    @staticmethod
    def log_optimization_config(optimizer_name: str, config: Dict[str, Any]):
        """Log DSPy optimizer configuration."""
        mlflow.log_param("optimizer_type", optimizer_name)

        for key, value in config.items():
            mlflow.log_param(f"optimizer_{key}", value)

    @staticmethod
    def log_training_data_info(training_examples):
        """Log information about training data."""
        mlflow.log_param("training_examples_count", len(training_examples))

        # Log sample queries for reference
        sample_queries = [ex.query for ex in training_examples[:5]]
        mlflow.log_param("sample_training_queries", str(sample_queries))


# Convenience functions
def get_mlflow_manager(experiment_name: str = None) -> MLflowManager:
    """Get configured MLflow manager."""
    return MLflowManager(experiment_name)


def setup_mlflow_for_dspy():
    """Setup MLflow for DSPy optimization."""
    DSPyMLflowIntegration.setup_dspy_autolog()
