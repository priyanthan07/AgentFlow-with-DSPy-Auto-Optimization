# dspy_config.py - Configuration for DSPy optimization

import os
from dotenv import load_dotenv

load_dotenv()

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

# Optimization Configuration
OPTIMIZATION_CONFIG = {
    "bootstrap_demos": {"max_bootstrapped_demos": 6, "max_labeled_demos": 3},
    "mipro_settings": {
        "auto": "medium",  # light, medium, heavy
        "num_threads": 4,
    },
    "training_thresholds": {
        "min_examples_for_random_search": 20,
        "min_examples_for_mipro": 50,
    },
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "quality_weights": {
        "sources_weight": 0.3,
        "findings_weight": 0.4,
        "depth_weight": 0.3,
    },
    "depth_scores": {"SURFACE": 0.1, "MODERATE": 0.2, "DEEP": 0.3},
    "target_metrics": {"min_sources": 3, "min_findings": 5, "min_quality_score": 0.7},
}

TRAINING_QUERIES = [
    # Technology Analysis
    "transformer architecture attention mechanisms computational efficiency comparison study",
    "quantum computing error correction surface codes implementation challenges 2025",
    "edge AI inference optimization techniques hardware acceleration methods",
    "federated learning privacy preservation techniques distributed systems",
    "neuromorphic computing architectures energy efficiency compared traditional processors",
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
    "healthcare AI regulation FDA approval processes medical device software"
]

# Test queries (different from training)
TEST_QUERIES = [
    "large language model fine-tuning techniques parameter efficient methods comparison",
    "artificial intelligence chip market competition Nvidia AMD Intel strategies",
    "solid-state battery technology lithium metal anodes commercial development timeline",
    "autonomous vehicle liability insurance regulatory frameworks international approaches",
    "containerization technologies Docker Kubernetes security performance comparison"
]
