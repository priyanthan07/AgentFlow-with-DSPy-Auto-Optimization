from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.agents.dspy_web_agent import WebResearchResult
from src.utils.logger import get_logger
from src.config import EVALUATION_CRITERIA

logger = get_logger(__name__)


@dataclass
class MetricResult:
    """Container for metric evaluation results."""

    score: float
    breakdown: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class GroundTruthCriteria:
    """Ground truth criteria for evaluation."""

    expected_topics: List[str]
    min_sources: int
    min_findings: int
    domain: str


class ResearchMetrics:
    """Hybrid evaluation combining ground truth validation with self-reported quality metrics."""

    def __init__(self):
        self.weights = {"ground_truth": 0.5, "self_reported": 0.5}

        self.ground_weight = {
            "topic_coverage": 0.4,
            "source_adequacy": 0.3,
            "findings_quality": 0.3,
        }

    def evaluate_research_quality(self, example, pred, trace=None) -> float:
        """
        Main DSPy metric that combines ground truth validation with self-reported quality.
        """
        try:
            if not hasattr(pred, "key_findings") or not hasattr(
                pred, "sources_analyzed"
            ):
                logger.warning("Prediction missing required attributes")
                return 0.0

            # Get query from example
            query = getattr(example, "query", "").lower()

            # Try ground truth evaluation first
            ground_truth_score = self._calculate_ground_truth_score(query, pred)

            # Calculate self-reported quality score
            self_reported_score = self._calculate_self_reported_score(pred)

            # Combine scores based on availability
            if ground_truth_score is not None:
                # We have ground truth - use hybrid approach
                final_score = (
                    ground_truth_score * self.weights["ground_truth"]
                    + self_reported_score * self.weights["self_reported"]
                )
                logger.debug(
                    f"Hybrid score: GT={ground_truth_score:.3f}, Self={self_reported_score:.3f}, Final={final_score:.3f}"
                )
            else:
                # No ground truth available - use self-reported with penalty
                final_score = (
                    self_reported_score * 0.8
                )  # 20% penalty for no external validation
                logger.debug(f"Self-reported only score: {final_score:.3f} (penalized)")

            return min(final_score, 1.0)

        except Exception as e:
            logger.error(f"Error in hybrid metric: {e}")
            return 0.0

    def _calculate_ground_truth_score(
        self, query: str, pred: WebResearchResult
    ) -> Optional[float]:
        """Calculate ground truth score if criteria exist for this query."""

        criteria = None
        for key_phrase, crit in EVALUATION_CRITERIA.items():
            key_words = key_phrase.split()[:4]
            matches = sum(1 for word in key_words if word in query)

            if matches >= 2:
                criteria = crit
                logger.debug(f"Found criteria for query using key: {key_phrase}")
                break

        if not criteria:
            logger.warning(f"No evaluation criteria found for query: {query}")
            return None

        # Topic coverage score (0-1)
        topic_score = self._calculate_topic_coverage(
            pred.key_findings, criteria.expected_topics
        )

        # Source adequacy score (0-1)
        source_score = min(pred.sources_analyzed / max(criteria.min_sources, 1), 1.0)

        # Findings quality score (0-1)
        findings_score = min(
            len(pred.key_findings) / max(criteria.min_findings, 1), 1.0
        )

        # Weighted final score
        final_score = (
            topic_score * self.ground_weight["topic_coverage"]
            + source_score * self.ground_weight["source_adequacy"]
            + findings_score * self.ground_weight["findings_quality"]
        )
        return final_score

    def _calculate_topic_coverage(
        self, findings: List[str], expected_topics: List[str]
    ) -> float:
        """Calculate how well findings cover expected topics."""

        if not expected_topics or not findings:
            return 0.0

        findings_text = " ".join(findings).lower()

        covered_topics = 0
        for topic in expected_topics:
            topic_words = topic.lower().split()
            matching_words = sum(1 for word in topic_words if word in findings_text)

            if len(topic_words) > 0 and matching_words / len(topic_words) >= 0.5:
                covered_topics += 1
                logger.debug(
                    f"Topic '{topic}' covered with {matching_words}/{len(topic_words)} words"
                )

        coverage_ratio = covered_topics / len(expected_topics)
        logger.debug(
            f"Topic coverage: {covered_topics}/{len(expected_topics)} = {coverage_ratio:.3f}"
        )
        return coverage_ratio

    def _calculate_self_reported_score(self, pred: WebResearchResult) -> float:
        """Calculate quality score based on self-reported metrics."""

        # Source diversity score (0.0 - 1.0)
        sources_score = min(pred.sources_analyzed / 7.0, 1.0)

        # Key findings score (0.0 - 1.0)
        findings_score = min(len(pred.key_findings) / 8.0, 1.0)

        # Research depth score (0.0 - 1.0)
        depth_scores = {"SURFACE": 0.4, "MODERATE": 0.7, "DEEP": 1.0}
        depth_score = depth_scores.get(pred.research_depth, 0.4)

        # Efficiency penalty - too many iterations suggests poor planning
        iterations = len(pred.react_trace) if pred.react_trace else 1
        efficiency_penalty = max(
            0.0, (iterations - 3) * 0.05
        )  # Small penalty for >3 iterations

        # Weighted score
        raw_score = sources_score * 0.3 + findings_score * 0.4 + depth_score * 0.3
        final_score = max(0.0, raw_score - efficiency_penalty)

        return final_score


# Export main metric function for DSPy
def research_quality_metric(example, pred, trace=None) -> float:
    """Main metric function for DSPy optimization."""
    evaluator = ResearchMetrics()
    return evaluator.evaluate_research_quality(example, pred, trace)
