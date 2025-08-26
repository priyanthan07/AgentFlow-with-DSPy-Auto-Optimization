import dspy
import time
import os
import mlflow
from typing import List
from src.agents.dspy_web_agent import DSPyWebResearchAgent
from src.optimization.metrics import research_quality_metric
from src.config import OPENAI_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DSPyOptimizer:

    def optimize_agent(self, agent: DSPyWebResearchAgent, training_examples: List[dspy.Example]):
        """
            Optimize DSPy agent with progressive optimization.
        """
        mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
        
        # Stage 1: Bootstrap optimization
        bootstrap_optimizer = dspy.BootstrapFewShot(
            metric=research_quality_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=1,
        )

        logger.info("Starting BootstrapFewShot optimization...")
        optimized_agent = bootstrap_optimizer.compile(agent, trainset=training_examples)

        # Stage 2: Advanced optimization if enough data
        logger.info("Applying MIPROv2 optimization.")
        mipro_optimization = dspy.MIPROv2(
            metric=research_quality_metric,
            auto="light",
            num_threads=1,
            num_trials=8,
            minibatch=True,
            minibatch_size=3,
            max_bootstrapped_demos=2,
            max_labeled_demos=1,
        )

        optimized_agent = mipro_optimization.compile(optimized_agent, trainset=training_examples)
        os.makedirs("src/optimized_agents", exist_ok=True)

        # Use consistent filename
        timestamp = int(time.time())
        agent_path = f"src/optimized_agents/dspy_optimized_web_agent_{timestamp}.json"

        # Save optimized agents
        try:
            optimized_agent.save(agent_path)
            mlflow.log_artifact(agent_path)
            logger.info(f"Optimized agent saved to {agent_path}")
            
        except Exception as e:
            logger.error(f"Failed to save optimized agent: {e}")

        logger.info("DSPy agent optimization completed")
        return optimized_agent
