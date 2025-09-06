# dspy_config.py - Configuration for DSPy optimization

import os
from dotenv import load_dotenv
from typing import List
from dataclasses import dataclass

load_dotenv()


@dataclass
class GroundTruthCriteria:
    """Ground truth criteria for evaluation."""

    expected_topics: List[str]
    min_sources: int
    min_findings: int
    domain: str


OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "default_model": "gpt-4o-2024-08-06",
    "embd_model": "text-embedding-3-small",
    "temperature": 0.1,
}

MCP_CONFIG = {
    "servers": {
        "web_research": {"url": os.getenv("MCP_WEB_RESEARCH_URL")},
        "arxiv_research": {"url": os.getenv("MCP_ARXIV_RESEARCH_URL")},
        "multimodal_analysis": {"url": os.getenv("MCP_MULTIMODAL_ANALYSIS_URL")},
    },
    "default_server": "web_research",
    "connection_timeout": 30,
    "retry_attempts": 3,
}

TAVILY_CONFIG = {"api_key": os.getenv("TAVILY_API_KEY")}

# DSPy Configuration
DSPY_CONFIG = {
    "model": f"openai/{OPENAI_CONFIG['default_model']}",
    "api_key": OPENAI_CONFIG["api_key"],
    "temperature": 0.1,
    "max_tokens": 1000,
}

# MLflow Configuration
MLFLOW_CONFIG = {
    "tracking_uri": "file:./mlruns",  # Adjust to your MLflow server
    "experiment_name": "DSPy_Web_Agent_Optimization",
    "artifact_location": "./mlflow_artifacts",
}

TRAINING_QUERIES = [
    # Technology Analysis
    "large language model fine-tuning techniques parameter efficient methods comparison",
    "edge AI inference optimization techniques hardware acceleration methods",
    "federated learning privacy preservation techniques distributed systems",
    "blockchain consensus mechanisms proof-of-stake versus proof-of-work analysis",
    "computer vision real-time object detection YOLO architecture improvements",
    "natural language processing large language models parameter scaling effects",
    # Business & Market Analysis
    "renewable energy storage market growth projections lithium-ion alternatives",
    "electric vehicle battery technology cost reduction trends manufacturing scale",
    "artificial intelligence enterprise adoption ROI measurement methodologies",
    "cloud computing infrastructure costs comparison AWS Azure Google Cloud",
    "cybersecurity market trends zero trust architecture implementation rates",
    "sustainable manufacturing automation environmental impact reduction strategies",
    "digital transformation small medium enterprises technology adoption barriers",
    "supply chain resilience artificial intelligence predictive analytics benefits",
    # Scientific Research
    "CRISPR gene editing therapeutic applications clinical trial success rates",
    "machine learning drug discovery pharmaceutical development timeline acceleration",
    "carbon capture technology direct air capture efficiency cost analysis",
    "solar panel efficiency perovskite tandem cells commercial viability",
    "fusion energy tokamak reactor designs ITER project progress assessment",
    "synthetic biology applications biofuel production environmental sustainability",
    "nanotechnology medical applications targeted drug delivery system effectiveness",
    "stem cell therapy regenerative medicine clinical applications regulatory approval",
    # Policy & Regulation
    "artificial intelligence governance frameworks EU AI Act implementation impact",
    "data privacy regulations GDPR compliance cost-benefit analysis businesses",
    "renewable energy policy incentives solar wind deployment acceleration",
    "autonomous vehicle regulation safety standards international comparison",
    "cryptocurrency regulation central bank digital currencies adoption strategies",
    "healthcare AI regulation FDA approval processes medical device software",
]

# Test queries (different from training)
TEST_QUERIES = [
    "transformer architecture attention mechanisms computational efficiency comparison study",
    "quantum computing error correction surface codes implementation challenges 2025",
    "artificial intelligence chip market competition Nvidia AMD Intel strategies",
    "solid-state battery technology lithium metal anodes commercial development timeline",
    "neuromorphic computing architectures energy efficiency compared traditional processors",
]

VALIDATION_QUERIES = [
    # Different technology domains
    "5G network infrastructure deployment challenges urban rural connectivity divide",
    "robotic process automation enterprise workflow optimization implementation strategies",
    "quantum cryptography post-quantum security algorithm standardization timeline",
    "augmented reality hardware development Apple Vision Pro market competition",
    "semiconductor manufacturing TSMC Intel foundry capacity expansion plans",
    # Different business areas
    "fintech blockchain payments cryptocurrency regulatory compliance frameworks",
    "digital transformation small medium enterprises technology adoption barriers",
    "clean energy investment trends venture capital funding patterns analysis",
    "supply chain automation artificial intelligence logistics optimization benefits",
    "sustainable manufacturing environmental impact reduction automation strategies",
    # Different scientific domains
    "gene therapy clinical trials FDA approval regulatory pathway challenges",
    "space technology commercial satellite deployment constellation management",
    "materials science graphene applications industrial manufacturing scalability",
]

EVALUATION_CRITERIA = {
    # Test queries criteria
    "transformer architecture attention mechanisms computational efficiency": GroundTruthCriteria(
        expected_topics=[
            "attention complexity",
            "computational efficiency",
            "transformer optimization",
            "hardware acceleration",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "quantum computing error correction surface codes implementation": GroundTruthCriteria(
        expected_topics=[
            "surface codes",
            "error correction",
            "quantum threshold",
            "implementation challenges",
        ],
        min_sources=4,
        min_findings=5,
        domain="technology",
    ),
    "artificial intelligence chip market competition Nvidia AMD": GroundTruthCriteria(
        expected_topics=[
            "nvidia strategy",
            "amd competition",
            "intel ai chips",
            "market share",
        ],
        min_sources=4,
        min_findings=5,
        domain="business",
    ),
    "solid-state battery technology lithium metal anodes commercial": GroundTruthCriteria(
        expected_topics=[
            "solid-state batteries",
            "lithium metal anodes",
            "commercial timeline",
            "development challenges",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "neuromorphic computing architectures energy efficiency compared": GroundTruthCriteria(
        expected_topics=[
            "neuromorphic computing",
            "energy efficiency",
            "traditional processors",
            "power consumption",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    # Training queries criteria
    "large language model fine-tuning techniques parameter efficient methods": GroundTruthCriteria(
        expected_topics=[
            "parameter efficient",
            "fine-tuning",
            "lora",
            "adapter methods",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "edge AI inference optimization techniques hardware acceleration": GroundTruthCriteria(
        expected_topics=[
            "edge computing",
            "inference optimization",
            "hardware acceleration",
            "mobile deployment",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "federated learning privacy preservation techniques distributed ": GroundTruthCriteria(
        expected_topics=[
            "federated learning",
            "privacy preservation",
            "distributed systems",
            "data security",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "blockchain consensus mechanisms proof-of-stake versus proof-of-work": GroundTruthCriteria(
        expected_topics=[
            "consensus mechanisms",
            "proof-of-stake",
            "proof-of-work",
            "energy efficiency",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "computer vision real-time object detection YOLO architecture": GroundTruthCriteria(
        expected_topics=[
            "computer vision",
            "YOLO",
            "real-time object detection",
            "alternative technologies",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
    "natural language processing large language models parameter scaling": GroundTruthCriteria(
        expected_topics=[
            "parameter scaling",
            "large language models",
            "model performance",
            "computational requirements",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "renewable energy storage market growth projections lithium-ion": GroundTruthCriteria(
        expected_topics=[
            "energy storage market",
            "lithium-ion alternatives",
            "market growth",
            "battery technology",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
    "electric vehicle battery technology cost reduction": GroundTruthCriteria(
        expected_topics=[
            "ev battery",
            "cost reduction",
            "manufacturing scale",
            "technology improvements",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
    "artificial intelligence enterprise adoption ROI measurement": GroundTruthCriteria(
        expected_topics=[
            "ai enterprise adoption",
            "roi measurement",
            "implementation costs",
            "business value",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
    "cloud computing infrastructure costs comparison AWS Azure": GroundTruthCriteria(
        expected_topics=[
            "cloud infrastructure",
            "cost comparison",
            "aws pricing",
            "azure google cloud",
        ],
        min_sources=4,
        min_findings=5,
        domain="business",
    ),
    "cybersecurity market trends zero trust architecture implementation": GroundTruthCriteria(
        expected_topics=[
            "cybersecurity market",
            "zero trust",
            "architecture implementation",
            "security trends",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
    "sustainable manufacturing automation environmental impact reduction": GroundTruthCriteria(
        expected_topics=[
            "sustainable manufacturing",
            "automation",
            "environmental impact",
            "reduction strategies",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
    "digital transformation small medium enterprises technology adoption": GroundTruthCriteria(
        expected_topics=[
            "digital transformation",
            "sme adoption",
            "technology barriers",
            "implementation challenges",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
    "supply chain resilience artificial intelligence predictive analytics": GroundTruthCriteria(
        expected_topics=[
            "supply chain resilience",
            "ai predictive analytics",
            "benefits",
            "implementation",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
    "CRISPR gene editing therapeutic applications clinical trial": GroundTruthCriteria(
        expected_topics=[
            "crispr gene editing",
            "therapeutic applications",
            "clinical trials",
            "success rates",
        ],
        min_sources=4,
        min_findings=5,
        domain="science",
    ),
    # Validation queries criteria
    "5G network infrastructure deployment challenges urban rural": GroundTruthCriteria(
        expected_topics=[
            "5G deployment",
            "infrastructure challenges",
            "urban rural divide",
            "connectivity",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "robotic process automation enterprise workflow optimization": GroundTruthCriteria(
        expected_topics=[
            "robotic process automation",
            "enterprise workflow",
            "optimization strategies",
            "implementation",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
    "quantum cryptography post-quantum security algorithm standardization": GroundTruthCriteria(
        expected_topics=[
            "quantum cryptography",
            "post-quantum security",
            "algorithm standardization",
            "security threats",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "augmented reality hardware development Apple Vision Pro market": GroundTruthCriteria(
        expected_topics=[
            "augmented reality",
            "hardware development",
            "market competition",
            "apple vision pro",
        ],
        min_sources=3,
        min_findings=4,
        domain="technology",
    ),
    "semiconductor manufacturing TSMC Intel foundry capacity expansion": GroundTruthCriteria(
        expected_topics=[
            "semiconductor manufacturing",
            "tsmc",
            "intel foundry",
            "capacity expansion",
        ],
        min_sources=3,
        min_findings=4,
        domain="business",
    ),
}
