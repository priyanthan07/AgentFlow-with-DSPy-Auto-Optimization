import dspy
import time
import os
import mlflow
import asyncio
from typing import List, Dict
from src.agents.dspy_web_agent import DSPyWebResearchAgent
from src.optimization.metrics import research_quality_metric
from src.config import OPENAI_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DSPyOptimizer:
    """DSPy optimizer with optimization tracking and training metrics."""

    def __init__(self):
        self.optimization_history = []
        self.best_optimization_score = 0.0

    def optimize_agent(
        self,
        agent: DSPyWebResearchAgent,
        training_examples: List[dspy.Example],
        validation_examples: List[dspy.Example] = None,
    ):
        """
        Optimize DSPy agent with progressive optimization.
        """

        mlflow.dspy.autolog(
            log_compiles=True, log_evals=True, log_traces_from_compile=True
        )
        logger.info("Starting DSPy optimization with tracking...")

        mlflow.log_params(
            {
                "training_examples": len(training_examples),
                "validation_examples": len(validation_examples),
                "optimizer_type": "Bootstrap + MIPROv2",
            }
        )

        # Stage 1: Baseline evaluation
        baseline_val_score = self._evaluate_on_dataset(agent, validation_examples[:3])

        mlflow.log_metrics({"baseline_val_score": baseline_val_score})

        logger.info(f"Baseline  Val: {baseline_val_score:.3f}")

        # Stage 2: Advanced optimization 
        logger.info("Applying MIPROv2 optimization.")
        mipro_optimization = dspy.MIPROv2(
            metric=self._tracked_metric, 
            auto="light", 
            num_threads=1,            
        )

        final_agent = mipro_optimization.compile(
            agent,
            trainset=training_examples,
            valset=validation_examples,
            minibatch=False,
        )

        # Save optimized agent
        self._save_optimized_agent(final_agent)

        logger.info("DSPy optimization completed")
        return final_agent

    def _tracked_metric(self, example, pred, trace=None):
        """Metric wrapper that tracks optimization trials."""
        score = research_quality_metric(example, pred, trace)

        mlflow.log_metrics({
            f"optimization_trial_score": score,
            f"optimization_trial_sources": getattr(pred, "sources_analyzed", 0),
            f"optimization_trial_findings": len(getattr(pred, "key_findings", [])),
        })
        
        # Track optimization trial
        trial_info = {
            "score": score,
            "sources_analyzed": getattr(pred, "sources_analyzed", 0),
            "findings_count": len(getattr(pred, "key_findings", [])),
            "timestamp": time.time(),
            "evaluation_method": "ground_truth",
        }
        self.optimization_history.append(trial_info)

        return score

    def _evaluate_on_dataset(
        self, agent: DSPyWebResearchAgent, examples: List[dspy.Example]
    ) -> float:
        """Evaluate agent on a subset of training examples."""
        scores = []

        for example in examples:
            try:
                result = asyncio.run(agent.research(example.query))
                score = research_quality_metric(example, result)
                scores.append(score)

            except Exception as e:
                logger.error(f"Error in training evaluation: {e}")
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _save_optimized_agent(self, agent: DSPyWebResearchAgent):
        """Save optimized agent with timestamp."""
        try:
            os.makedirs("src/optimized_agents", exist_ok=True)
            timestamp = int(time.time())
            agent_path = (
                f"src/optimized_agents/dspy_optimized_web_agent_{timestamp}.json"
            )

            agent.save(agent_path)
            mlflow.log_artifact(agent_path)
            logger.info(f"Optimized agent saved to {agent_path}")

        except Exception as e:
            logger.error(f"Failed to save optimized agent: {e}")
