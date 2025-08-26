import asyncio
import dspy
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.agents.web_agent import WebResearchAgent
from src.agents.dspy_web_agent import DSPyWebResearchAgent
from src.optimization.optimizer import DSPyOptimizer
from src.optimization.evaluator import AgentEvaluator
from src.utils.mlflow_utils import MLflowManager
from src.utils.logger import get_logger
from src.config import TRAINING_QUERIES, TEST_QUERIES

logger = get_logger(__name__)

class DataProvider:
    """Simple data provider using curated queries. """

    @staticmethod
    def get_training_examples(count: int = 30) -> List[dspy.Example]:
        queries = TRAINING_QUERIES[:count]

        training_examples = []
        for query in queries:
            example = dspy.Example(query=query).with_inputs("query")
            training_examples.append(example)

        logger.info(f" Created {len(training_examples)} curated training examples")
        return training_examples
    
    @staticmethod
    def get_test_queries(count: int = 5) -> List[str]:
        return TEST_QUERIES[:count]


class ExperimentRunner:
    def __init__(self):
        self.optimizer = DSPyOptimizer()
        self.evaluator = AgentEvaluator()
        self.mlflow_manager = MLflowManager()
        self.data_provider = DataProvider()
    
    async def run_full_experiment(
        self, 
        training_size: int = 10,
        test_size: int = 3
    ):
        """Run complete optimization and evaluation experiment."""
        logger.info(" Starting DSPy Optimization Experiment")

        experiment_config = {
            "training_size": training_size,
            "test_size": test_size,
        }
        
        with self.mlflow_manager.start_optimization_run("Experiment", experiment_config):
            # Step 1: Get data 
            training_examples = self.data_provider.get_training_examples(training_size)
            test_queries = self.data_provider.get_test_queries(test_size)

            logger.info(f" Training examples: {len(training_examples)} ")
            logger.info(f" Test queries: {len(test_queries)} ")
        
            # Step 2: Initialize agents
            original_agent = await WebResearchAgent.create()
            dspy_agent = await DSPyWebResearchAgent.create()
        
            # Step 3: Optimize DSPy agent
            optimized_agent = self.optimizer.optimize_agent(dspy_agent, training_examples)
        
            # Step 4: Compare agents
            results, improvements = await self.evaluator.compare_agents(
                original_agent, optimized_agent, test_queries
            )

            # Generate summary
            summary = self._generate_experiment_summary(
                results, improvements, experiment_config, training_examples, test_queries
            )
            
            logger.info("Experiment completed successfully!")
            return {
                "summary": summary,
                "improvements": improvements,
                "results": results,
                "config": experiment_config
            }
        
    def _generate_experiment_summary(
            self,
            results: Dict[str, List],
            improvements: Dict[str, float],
            config: Dict[str, Any],
            training_examples: List[dspy.Example],
            test_queries: List[str]
    ) -> Dict[str, Any]:
        """Generate experiment summary."""

        original_avg_quality = sum(result["quality_score"] for result in results["original"])/ len(results["original"])
        dspy_avg_quality = sum(result["quality_score"] for result in results["dspy"]) / len(results["dspy"])

        overall_improvement = sum(improvements.values()) / len(improvements)

        return {
            "experiment_approach": "curated_data",
            "training_data_info": {
                "approach": "curated_high_quality_examples",
                "training_count": len(training_examples),
                "test_count": len(test_queries),
                "data_quality": "pre_validated_research_queries",
                "no_generation_needed": True
            },
            "optimization_results": {
                "original_avg_quality": round(original_avg_quality, 3),
                "dspy_avg_quality": round(dspy_avg_quality, 3),
                "overall_improvement_percent": round(overall_improvement, 2),
                "individual_improvements": improvements
            },
            "experiment_benefits": [
                "no_complex_data_generation",
                "single_dataset_optimizes_all_modules",
                "end_to_end_optimization",
                "faster_execution",
                "cleaner_codebase"
            ],
            "config": config
        }
    