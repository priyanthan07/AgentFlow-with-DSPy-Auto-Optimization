import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.agents.dspy_web_agent import WebResearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    score: float
    breakdown: Dict[str, float]
    metadata : Dict[str, Any]

class ResearchQualityMetrics:
    """Comprehensive research quality evaluation metrics."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        if weights:
            self.weights = weights
        else:
            self.weights = {
                "sources" : 0.25,
                "findings" : 0.35,
                "depth" : 0.25
            }
            
    def research_quality_metric(self, example, pred, trace=None) -> float:
        """Main DSPy metric function for optimization."""
        try:
            if not hasattr(pred, "sources_analyzed") or not hasattr(pred, "key_findings"):
                return 0.0
            
            result = self.comprehensive_evaluation(pred)
            return result.score
        
        except Exception as e:
            logger.error(f"Error in quality metric: {e}")
            return 0.0

    def comprehensive_evaluation(self, result: WebResearchResult) -> MetricResult:
        """Comprehensive evaluation with detailed breakdown."""

        # Source diversity score (0.0 - 1.0)
        sources_score = min(result.sources_analyzed / 5.0, 1.0)   # 5 is more than enough

        # Key findings quality (0.0 - 1.0)
        findings_score = min(len(result.key_findings) / 8.0, 1.0) # 8 is more than enough

        # Research depth (0-0.3)
        depth_scores = {"SURFACE": 0.4, "MODERATE": 0.7, "DEEP": 1.0}
        depth_score = depth_scores.get(result.research_depth, 0.0)

        # Weighted total score
        total_score = (
            sources_score * self.weights["sources"] +
            findings_score * self.weights["findings"] +
            depth_score * self.weights["depth"]
        )

        breakdown = {
            "sources_score": sources_score,
            "findings_score": findings_score,
            "depth_score": depth_score,
            "weighted_total": total_score
        }

        metadata = {
            "sources_analyzed": result.sources_analyzed,
            "findings_count": len(result.key_findings),
            "research_depth": result.research_depth,
            "iterations": len(result.react_trace) if result.react_trace else 0
        }

        return MetricResult(
            score=total_score,
            breakdown=breakdown,
            metadata=metadata
        )
    
class PerformanceMetrics:
    """Performance and efficiency metrics."""

    @staticmethod
    def efficiency_score(result: WebResearchResult) -> float:
        """Calculate efficiency based on iterations and output quality."""

        max_iterations = 5
        iterations = len(result.react_trace) if result.react_trace else 1

        # Efficiency = quality / iterations (normalized)
        quality = ResearchQualityMetrics().research_quality_metric(None, result)
        efficiency = quality / (iterations/max_iterations)

        return min(efficiency, 1.0)
    
    @staticmethod
    def consistency_score(results: List[WebResearchResult]) -> float:
        # Efficiency = quality / iterations (normalized)
        if len(results) < 2:
            return 1.0
        
        scores = [ResearchQualityMetrics().research_quality_metric(None, result) for result in results]
        std_dev = statistics.stdev(scores)

        # Lower standard deviation = higher consistency
        # Normalize to 0-1 scale (assuming max std_dev of 0.5)
        consistency = max(0.0, 1.0 - (std_dev/0.5))
        return consistency
    
class DomainSpecificMetrics:
    """Domain-specific evaluation metrics."""

    @staticmethod
    def technical_accuracy_metric(example, pred, trace=None) -> float:
        """Metric optimized for technical queries."""
        base_score = ResearchQualityMetrics().research_quality_metric(example, pred, trace)

        # Bonus for technical depth
        if hasattr(pred, 'research_depth') and pred.research_depth == "DEEP":
            base_score *=1.2

        # Bonus for multiple technical sources
        if hasattr(pred, 'sources_analyzed') and pred.sources_analyzed >= 4:
            base_score *= 1.1

        return min(base_score, 1.0)

    @staticmethod
    def speed_optimized_metric(example, pred, trace=None) -> float:
        """Metric that balances quality with speed."""

        quality_score = ResearchQualityMetrics().research_quality_metric(example, pred, trace)

        # penalty for too many iterations
        iterations = len(pred.react_trace) if hasattr(pred, 'react_trace') and pred.react_trace else 1
        speed_penalty = max(0.0, (iterations-2) * 0.1)
        
        final_score = quality_score - speed_penalty
        return max(0.0, final_score)
    
# Export main metric function for DSPy
def research_quality_metric(example, pred, trace=None) -> float:
    """Main metric function for DSPy optimization."""
    return ResearchQualityMetrics().research_quality_metric(example, pred, trace)
