import mlflow
import time
from typing import Dict, List, Any, Tuple
from src.agents.web_agent import WebResearchAgent
from src.agents.dspy_web_agent import DSPyWebResearchAgent
from src.optimization.metrics import research_quality_metric
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AgentEvaluator:
    """Comprehensive agent evaluation with quality-focused metrics."""

    async def compare_agents(self, original_agent: WebResearchAgent, dspy_agent: DSPyWebResearchAgent, test_queries: List[str]) -> tuple[Dict[str, List], Dict[str, float]]:
        """
            Compare performance of original and DSPy agents.
        """
        results = {"original":[], "dspy":[]}

        for query in test_queries:
            logger.info(f"Evaluating query: {query}")

            # Test original agent
            with mlflow.start_run(run_name=f"Original_agent_{query[:20]}", nested=True):
                start_time = time.time()
                original_result = await original_agent.research(query)
                original_time = time.time() - start_time

                original_metrics = self._calculate_detailed_metrics(original_result, original_time)
                results["original"].append(original_metrics)

                mlflow.log_param("query", query)
                mlflow.log_param("agent_type", "original")
                for key, value in original_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)

            # Test DSPy agent
            with mlflow.start_run(run_name=f"DSPy_Agent_{query[:20]}", nested=True):
                start_time = time.time()
                dspy_result = await dspy_agent.research(query)
                dspy_time = time.time() - start_time

                dspy_metrics = self._calculate_detailed_metrics(dspy_result, dspy_time)
                results["dspy"].append(dspy_metrics)

                # Log to MLflow
                mlflow.log_param("query", query)
                mlflow.log_param("agent_type", "dspy_optimized")
                for key, value in dspy_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
            
        # Calculate improvements
        improvements = self._calculate_improvements(results)
        
        with mlflow.start_run(run_name="Agent_Comparison_Summary", nested=True):
            for metric, improvement in improvements.items():
                mlflow.log_metric(f"improvement_{metric}", improvement)

            # Log overall assessment
            overall_improvement = sum(improvements.values()) / len(improvements)
            mlflow.log_metric("overall_improvement_percent", overall_improvement)
            
            # Log success criteria
            mlflow.log_metric("quality_improved", 1 if improvements.get("quality_score", 0) > 0 else 0)
            mlflow.log_metric("efficiency_improved", 1 if improvements.get("efficiency_score", 0) > 0 else 0)

        logger.info(f"Comparison completed. Improvements: {improvements}")
        return results, improvements

    def _calculate_detailed_metrics(self, result, exe_time: float) -> Dict[str, Any]:

        # Basic metrics
        base_metrics = {
            "query": result.query,
            "sources_analyzed": result.sources_analyzed,
            "key_findings_count": len(result.key_findings),
            "research_depth": result.research_depth,
            "iterations": len(result.react_trace),
            "execution_time_seconds": round(exe_time, 2)
        }

        quality_score = research_quality_metric(None, result)
 
        efficiency_score = quality_score / max(exe_time/60, 0.1)  # Quality per minute
        iteration_efficiency= quality_score / max(len(result.react_trace), 1) # Quality per iteration

        # Research completeness assessment
        completeness_score = self._assess_completeness(result)

        # Source relevance assessment
        relevance_score = self._assess_source_relevance(result)

        comprehensive_metrics ={
            **base_metrics,
            "quality_score" : round(quality_score, 3),
            "efficiency_score": round(efficiency_score, 3),
            "iteration_efficiency": round(iteration_efficiency, 3),
            "completeness_score": round(completeness_score, 3),
            "relevance_score": round(relevance_score, 3),
            "overall_research_score": round((quality_score + completeness_score + relevance_score) / 3, 3)
        }

        return comprehensive_metrics
    
    def _assess_completeness(self, result) -> float:

        completeness_factors = []

        # Source diversity (0.0 - 1.0)
        source_diversity = min(result.sources_analyzed / 4.0, 1.0)  # Target: 4+ sources
        completeness_factors.append(source_diversity)

        # FInding richness (0.0 - 1.0) 
        finding_richness = min(len(result.key_findings)/6.0, 1.0)   # Target: 6+ findings
        completeness_factors.append(finding_richness)

        # Research depth appropriateness (0.0 - 1.0)
        depth_scores = {"SURFACE": 0.4, "MODERATE": 0.7, "DEEP": 1.0}
        depth_score = depth_scores.get(result.research_depth, 0.0)
        completeness_factors.append(depth_score)

        # Summary quality
        summary_length_score = min(len(result.summary.split()) / 200, 1.0)  # Target: 200+ words
        completeness_factors.append(summary_length_score)

        return sum(completeness_factors) / len(completeness_factors)
    
    def _assess_source_relevance(self, result) -> float:

        if result.sources_analyzed == 0:
            return 0.0
        
        # Ratio of sources analyzed to sources found
        sources_found = len(result.search_results)
        if sources_found == 0:
            return 0.5  # Default score if no search results
        
        analysis_ratio = result.sources_analyzed / sources_found

        # Findings per source (indicates relevance)
        findings_per_source = len(result.key_findings) / max(result.sources_analyzed, 1)
        findings_score = min(findings_per_source / 2.0, 1.0)  # Target: 2+ findings per source
        
        # Combine factors
        relevance_score = (analysis_ratio + findings_score) / 2

        return min(relevance_score, 1.0)
    
    def _calculate_improvements(self, results: Dict) -> Dict[str, float]:
        
        improvements = {}

        # Define metrics to compare
        metrics_to_compare = [
            "quality_score", "efficiency_score", "iteration_efficiency",
            "completeness_score", "relevance_score", "overall_research_score",
            "sources_analyzed", "key_findings_count"
        ]

        for metric in metrics_to_compare:
            original_values = [result[metric] for result in results["original"] if metric in result]
            dspy_values = [result[metric] for result in results["dspy"] if metric in result]

            if original_values and dspy_values:
                original_avg = sum(original_values) / len(original_values)
                dspy_avg = sum(dspy_values) / len(dspy_values)

                if original_avg > 0:
                    improvement = ((dspy_avg - original_avg) / original_avg) * 100
                    improvements[metric] = round(improvement, 2)
                else:
                    improvements[metric] = 0.0

        return improvements
