import mlflow
import time
import json
from typing import Dict, List, Any, Tuple
from collections import Counter
from src.agents.web_agent import WebResearchAgent
from src.agents.dspy_web_agent import DSPyWebResearchAgent
from src.optimization.metrics import research_quality_metric
from src.utils.logger import get_logger
from dataclasses import dataclass
from src.config import EVALUATION_CRITERIA

logger = get_logger(__name__)


@dataclass
class EvaluationExample:
    query: str


class AgentEvaluator:
    """Agent evaluator with separated quality and speed metrics to avoid bias."""

    async def compare_agents(
        self,
        original_agent: WebResearchAgent,
        dspy_agent: DSPyWebResearchAgent,
        test_queries: List[str],
    ) -> Tuple[Dict[str, List], Dict[str, float]]:
        """Compare agents with separated quality and speed metrics."""

        test_data = []
        for query in test_queries:
            if self._has_evaluation_criteria(query):
                test_data.append(query)
            else:
                logger.warning(f"No evaluation criteria for query: {query}")

        if not test_data:
            logger.warning("No test queries have evaluation criteria defined")
            test_data = test_queries

        logger.info(
            f"Evaluating {len(test_data)} queries with ground truth criteria"
        )

        results = {"original": [], "dspy": []}
        search_patterns = {"original": [], "dspy": []}

        for query_idx, query in enumerate(test_data):
            logger.info(
                f"Evaluating query {query_idx + 1}/{len(test_data)}: {query}"
            )

            # Test original agent
            (
                original_metrics,
                original_searches,
            ) = await self._evaluate_agent_with_ground_truth(
                original_agent, query, "original"
            )
            results["original"].append(original_metrics)
            search_patterns["original"].extend(original_searches)

            # Log directly to current run
            mlflow.log_metrics(
                {
                    f"original_q{query_idx + 1}_quality_score": original_metrics[
                        "quality_score"
                    ],
                    f"original_q{query_idx + 1}_execution_time": original_metrics[
                        "execution_time_seconds"
                    ],
                    f"original_q{query_idx + 1}_sources": original_metrics[
                        "sources_analyzed"
                    ],
                }
            )

            # Test DSPy agent -
            dspy_metrics, dspy_searches = await self._evaluate_agent_with_ground_truth(
                dspy_agent, query, "dspy"
            )
            results["dspy"].append(dspy_metrics)
            search_patterns["dspy"].extend(dspy_searches)

            # Log directly to current run
            mlflow.log_metrics(
                {
                    f"dspy_q{query_idx + 1}_quality_score": dspy_metrics[
                        "quality_score"
                    ],
                    f"dspy_q{query_idx + 1}_execution_time": dspy_metrics[
                        "execution_time_seconds"
                    ],
                    f"dspy_q{query_idx + 1}_sources": dspy_metrics["sources_analyzed"],
                }
            )

        # Calculate improvements - SEPARATED metrics
        improvements = self._calculate_separated_improvements(results)

        # Analyze search diversity
        search_analysis = self._analyze_search_diversity(search_patterns)

        # Log all results to current MLflow run (no nested runs)
        mlflow.log_metrics(improvements)
        mlflow.log_metrics(search_analysis)

        # Log search pattern analysis
        self._log_evaluation_analysis(results)

        logger.info(f"Comparison completed. Quality improvements: {improvements}")
        return results, improvements

    def _has_evaluation_criteria(self, query: str) -> bool:
        """Check if query has ground truth evaluation criteria."""
        query_lower = query.lower()
        for key_phrase in EVALUATION_CRITERIA.keys():
            key_words = key_phrase.split()[:4]
            matches = sum(1 for word in key_words if word in query_lower)
            if matches >= 2:
                return True
        return False

    async def _evaluate_agent_with_ground_truth(
        self, agent, query: str, agent_type: str
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Simple agent evaluation with separated metrics."""

        start_time = time.time()

        try:
            result = await agent.research(query)
            execution_time = time.time() - start_time

            example = EvaluationExample(query=query)

            quality_score = research_quality_metric(example, result)

            # Extract search queries for diversity analysis
            search_queries = self._extract_search_queries(result)

            metrics = {
                "query": query,
                "agent_type": agent_type,
                # Quality metrics (time-independent)
                "quality_score": round(quality_score, 4),
                "sources_analyzed": result.sources_analyzed,
                "key_findings_count": len(result.key_findings),
                "research_depth": result.research_depth,
                # Speed metrics (separated from quality)
                "execution_time_seconds": round(execution_time, 2),
                "iterations": len(result.react_trace) if result.react_trace else 0,
                # Process metrics
                "search_queries_count": len(search_queries),
                "unique_search_queries": len(set(search_queries)),
            }

            return metrics, search_queries

        except Exception as e:
            logger.error(f"Error evaluating {agent_type} agent: {e}")
            return {
                "query": query,
                "agent_type": agent_type,
                "quality_score": 0.0,
                "sources_analyzed": 0,
                "key_findings_count": 0,
                "research_depth": "SURFACE",
                "execution_time_seconds": 0.0,
                "iterations": 0,
                "search_queries_count": 0,
                "unique_search_queries": 0,
                "error": str(e),
            }, []

    def _extract_search_queries(self, result) -> List[str]:
        """Extract search queries from research result."""
        search_queries = []

        if hasattr(result, "react_trace") and result.react_trace:
            for step in result.react_trace:
                if "web_search" in step.action.lower() and step.action_params:
                    query = step.action_params.get("query", "")
                    if query:
                        search_queries.append(query.lower().strip())

        return search_queries

    def _calculate_separated_improvements(self, results: Dict) -> Dict[str, float]:
        """Calculate improvements with SEPARATED quality and speed metrics."""
        improvements = {}

        # Primary metric: Ground truth score improvement
        original_scores = [r["quality_score"] for r in results["original"]]
        dspy_scores = [r["quality_score"] for r in results["dspy"]]

        if original_scores and dspy_scores:
            original_avg = sum(original_scores) / len(original_scores)
            dspy_avg = sum(dspy_scores) / len(dspy_scores)

            if original_avg > 0:
                improvement = ((dspy_avg - original_avg) / original_avg) * 100
                improvements["ground_truth_score_improvement_percent"] = round(
                    improvement, 2
                )

            # Raw score comparison
            improvements["original_avg_ground_truth_score"] = round(original_avg, 4)
            improvements["dspy_avg_ground_truth_score"] = round(dspy_avg, 4)

        # Secondary metrics for analysis (not combined with quality)
        secondary_metrics = [
            "sources_analyzed",
            "key_findings_count",
            "execution_time_seconds",
            "iterations",
        ]

        for metric in secondary_metrics:
            original_values = [r[metric] for r in results["original"] if metric in r]
            dspy_values = [r[metric] for r in results["dspy"] if metric in r]

            if original_values and dspy_values:
                original_avg = sum(original_values) / len(original_values)
                dspy_avg = sum(dspy_values) / len(dspy_values)

                if original_avg > 0:
                    change = ((dspy_avg - original_avg) / original_avg) * 100
                    improvements[f"{metric}_change_percent"] = round(change, 2)

        return improvements

    def _analyze_search_diversity(
        self, search_patterns: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Analyze search query diversity patterns."""
        analysis = {}

        for agent_type, queries in search_patterns.items():
            if queries:
                unique_queries = len(set(queries))
                total_queries = len(queries)

                # Diversity score
                diversity_score = unique_queries / max(total_queries, 1)

                # Repetition analysis
                query_counts = Counter(queries)
                most_common = query_counts.most_common(1)
                max_repetition = most_common[0][1] if most_common else 0

                analysis[f"{agent_type}_search_diversity_score"] = round(
                    diversity_score, 4
                )
                analysis[f"{agent_type}_max_query_repetition"] = max_repetition
                analysis[f"{agent_type}_total_search_queries"] = total_queries
                analysis[f"{agent_type}_unique_search_queries"] = unique_queries

        return analysis

    def _log_evaluation_analysis(self, results: Dict[str, List]):
        """Log evaluation analysis to MLflow."""
        try:
            # Calculate summary statistics
            for agent_type, agent_results in results.items():
                if agent_results:
                    quality_scores = [r["quality_score"] for r in agent_results]
                    execution_times = [
                        r["execution_time_seconds"] for r in agent_results
                    ]
                    sources_counts = [r["sources_analyzed"] for r in agent_results]

                    # Log summary metrics
                    mlflow.log_metrics(
                        {
                            f"{agent_type}_avg_quality": sum(quality_scores)
                            / len(quality_scores),
                            f"{agent_type}_avg_execution_time": sum(execution_times)
                            / len(execution_times),
                            f"{agent_type}_avg_sources": sum(sources_counts)
                            / len(sources_counts),
                            f"{agent_type}_min_quality": min(quality_scores),
                            f"{agent_type}_max_quality": max(quality_scores),
                        }
                    )

                    # Log detailed results as artifact
                    results_file = f"{agent_type}_detailed_results.json"
                    with open(results_file, "w") as f:
                        json.dump(agent_results, f, indent=2)

                    mlflow.log_artifact(results_file)
                    import os

                    os.remove(results_file)  # Clean up temp file

        except Exception as e:
            logger.error(f"Error logging evaluation analysis: {e}")
