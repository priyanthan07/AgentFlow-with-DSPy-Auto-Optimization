import json
import dspy
import asyncio
import nest_asyncio
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI
import mlflow
from src.mcp_client.client import create_mcp_client
from src.utils.logger import get_logger
from src.config import OPENAI_CONFIG

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

logger = get_logger(__name__)


@dataclass
class ReActStep:  # Represents one complete ReAct cycle
    iteration: int
    thought: str
    action: str
    action_params: Dict[str, Any]
    action_results: Dict[str, Any]
    observation: str
    reflection: str
    timestamp: datetime = datetime.now()


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    content: str = ""
    source_type: str = "web"
    extracted_at: datetime = datetime.now()


@dataclass
class WebResearchResult:
    query: str
    search_results: List[SearchResult]
    summary: str
    key_findings: List[str]
    sources_analyzed: int
    research_depth: str  # "SURFACE", "MODERATE", "DEEP"
    react_trace: List[ReActStep]
    metadata: Dict[str, Any]


class WebResearchSignature(dspy.Signature):
    query: str = dspy.InputField(desc="The research question or query to investigate")
    final_answer: str = dspy.OutputField(
        desc="Comprehensive research summary that directly answers the question"
    )


class SynthesisSignature(dspy.Signature):
    original_query: str = dspy.InputField(
        desc="The research question that was investigated"
    )
    search_results_summary: str = dspy.InputField(
        desc="Summary of all search results and analyses"
    )
    key_findings: str = dspy.InputField(
        desc="List of key findings discovered during research"
    )
    summary: str = dspy.OutputField(
        desc="Comprehensive research summary that directly answers the question (300-500 words)"
    )


class FindingsExtractionSignature(dspy.Signature):
    content: str = dspy.InputField(desc="Content to analyze for key findings")
    research_query: str = dspy.InputField(
        desc="Research question to focus the extraction"
    )
    key_findings: str = dspy.OutputField(
        desc="3-5 specific, factual findings that directly relate to the query, one per line"
    )


class SynthesisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesisSignature)

    def forward(
        self, original_query: str, search_results_summary: str, key_findings: str
    ):
        return self.synthesize(
            original_query=original_query,
            search_results_summary=search_results_summary,
            key_findings=key_findings,
        )


class FindingsModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_findings = dspy.ChainOfThought(FindingsExtractionSignature)

    def forward(self, content: str, research_query: str):
        return self.extract_findings(content=content, research_query=research_query)


class DSPyWebResearchAgent(dspy.Module):
    """DSPy-optimized Web Research Agent with automatic prompt optimization."""

    def __init__(self):
        super().__init__()

        # Configure DSPy
        lm = dspy.LM(
            model=f"openai/{OPENAI_CONFIG['default_model']}",
            api_key=OPENAI_CONFIG["api_key"],
        )

        dspy.configure(lm=lm)

        # Apply nest_asyncio for event loop handling
        nest_asyncio.apply()

        # Initialize DSPy modules
        self.synthesis_module = SynthesisModule()
        self.finding_module = FindingsModule()

        self.client = None
        self.max_iterations = 3
        self.is_initialized = False

        self.available_tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information. Use when you need to find new sources or explore a topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results (1-20)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_webpage",
                    "description": "Analyze a webpage for detailed content. Use when you want to extract information from a specific URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to analyze"},
                            "extract_text": {"type": "boolean", "default": True},
                            "summarize": {"type": "boolean", "default": True},
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

        logger.info("DSPy Web Research Agent initialized successfully")

    def _ensure_openai_client(self):
        """Ensure OpenAI client exists and is properly initialized."""
        if self.client is None or not hasattr(self.client, "_state"):
            try:
                # Close existing client if it exists but is problematic
                if hasattr(self, "client") and self.client is not None:
                    try:
                        self.client.close()
                    except:
                        pass
            except:
                pass
            self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])

    async def _ensure_initialized(self):
        """Ensure the agent is properly initialized before use."""
        if not hasattr(self, "mcp_client") or not self.is_initialized:
            await self._initialize_mcp_connection()
            self.is_initialized = True

        # Also ensure OpenAI client exists
        self._ensure_openai_client()

    @classmethod
    async def create(cls) -> "DSPyWebResearchAgent":
        agent = cls()
        await agent._initialize_mcp_connection()
        agent.is_initialized = True
        return agent

    async def _initialize_mcp_connection(self):
        try:
            self.mcp_client = create_mcp_client()
            await self.mcp_client._initialize_client(server="web_research")

            server_tools = [tool["name"] for tool in self.mcp_client.available_tools]
            expected_tools = [tool["function"]["name"] for tool in self.available_tools]

            for expected_tool in expected_tools:
                if expected_tool not in server_tools:
                    logger.warning(
                        f"Expected tool '{expected_tool}' not available from MCP server"
                    )

            logger.info(
                f"MCP connection established. Available tools in server : {server_tools}"
            )

        except Exception as e:
            logger.error(f"Failed to connect with web mcp client: {e}")
            raise RuntimeError(e)

    def forward(self, query: str, context: List[Dict] = None) -> WebResearchResult:
        """DSPy forward method for optimization."""
        return asyncio.run(self.research(query, context))

    def reset_copy(self):
        """
        Create a fresh copy of the agent for DSPy optimization.
        This method is required by DSPy's BootstrapFewShot optimizer.
        """
        new_agent = self.__class__()

        new_agent.max_iterations = getattr(self, "max_iterations", 3)
        new_agent.is_initialized = False

        return new_agent

    def copy(self):
        """
        Alternative copy method that some DSPy optimizers might use.
        """
        return self.reset_copy()

    def __deepcopy__(self, memo):
        """Custom deepcopy to handle OpenAI client properly."""
        new_agent = self.__class__()
        new_agent.max_iterations = getattr(self, "max_iterations", 3)
        new_agent.is_initialized = False
        new_agent.client = None
        new_agent.mcp_client = None
        return new_agent

    def web_search_tool(self, query: str, num_results: int = 10) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._execute_web_search(query, num_results))

    def analyze_webpage_tool(
        self, url: str, extract_text: bool = True, summarize: bool = True
    ) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._execute_webpage_analysis(url, extract_text, summarize)
        )

    async def research(
        self, task_query: str, context: List[Dict] = None
    ) -> WebResearchResult:
        """Main research method with DSPy optimization."""

        logger.info(f"Starting DSPy web research for query: {task_query}")
        await self._ensure_initialized()

        # Clear previous trajectory data
        self.detailed_trajectory = []

        # Store current query for context
        self.current_query = task_query
        self.research_state = {
            "original_query": task_query,
            "context": context or [],
            "search_results": [],
            "key_findings": [],
            "analyzed_sources": [],
            "iteration": 0,
            "research_complete": False,
            "react_steps": [],
        }

        react_agent = dspy.ReAct(
            signature=WebResearchSignature,
            tools=[self.web_search_tool, self.analyze_webpage_tool],
            max_iters=self.max_iterations,
        )

        # Execute ReAct reasoning cycle
        logger.info("Executing DSPy ReAct reasoning cycle...")
        react_result = react_agent(query=task_query)

        # Log detailed trajectory
        self._log_dspy_trajectory(react_result)

        # Extract trajectory information for compatibility
        trajectory = getattr(react_result, "trajectory", {})
        react_steps = self._convert_trajectory_to_steps(trajectory)

        # Update research state for compatibility
        self.research_state["react_steps"] = react_steps

        # Synthesize final results
        final_result = await self._dspy_synthesize_results()

        logger.info(
            f"DSPy Web research completed with {len(final_result.search_results)} sources"
        )
        return final_result

    def _log_dspy_trajectory(self, react_result):
        """Log detailed DSPy ReAct trajectory for visibility."""

        if hasattr(react_result, "trajectory"):
            trajectory = react_result.trajectory

            logger.info("=== DSPy ReAct Trajectory ===")

            # Log each step in detail
            step_keys = [k for k in trajectory.keys() if k.startswith("thought_")]

            for i in range(len(step_keys)):
                thought = trajectory.get(f"thought_{i}", "No thought recorded")
                tool_name = trajectory.get(f"tool_name_{i}", "No tool")
                tool_args = trajectory.get(f"tool_args_{i}", {})
                observation = trajectory.get(f"observation_{i}", "No observation")

                logger.info(f"--- Step {i + 1} ---")
                logger.info(f"THOUGHT: {thought}")
                logger.info(f"ACTION: {tool_name}({tool_args})")
                logger.info(
                    f"OBSERVATION: {observation[:200]}..."
                    if len(str(observation)) > 200
                    else f"OBSERVATION: {observation}"
                )

                # Log to MLflow if available
                try:
                    mlflow.log_text(
                        f"DSPy Step {i+1} - Thought: {thought}",
                        f"dspy_step_{i+1}_thought.txt",
                    )
                    mlflow.log_text(
                        f"DSPy Step {i+1} - Action: {tool_name}({tool_args})",
                        f"dspy_step_{i+1}_action.txt",
                    )
                    mlflow.log_text(
                        f"DSPy Step {i+1} - Observation: {observation}",
                        f"dspy_step_{i+1}_observation.txt",
                    )
                except:
                    pass  # MLflow may not be active

            logger.info("=== End DSPy Trajectory ===")

    def _convert_trajectory_to_steps(self, trajectory: dict) -> List[ReActStep]:
        """Convert DSPy ReAct trajectory to our ReActStep format for compatibility."""
        react_steps = []

        if not trajectory:
            return react_steps

        # Extract steps from DSPy trajectory
        i = 0

        thought_keys = [k for k in trajectory.keys() if k.startswith("thought_")]

        while i < len(thought_keys):
            thought = trajectory.get(f"thought_{i}", "")
            tool_name = trajectory.get(f"tool_name_{i}", "")
            tool_args = trajectory.get(f"tool_args_{i}", {})
            observation = trajectory.get(f"observation_{i}", "")

            action_str = f"{tool_name}({tool_args})" if tool_name else "No action"

            react_step = ReActStep(
                iteration=i + 1,
                thought=thought,
                action=action_str,
                action_params=tool_args,
                action_results={"observation": observation},
                observation=observation,
                reflection="CONTINUE" if i < len(thought_keys) - 1 else "CONCLUDE",
            )
            react_steps.append(react_step)
            i += 1

        return react_steps

    async def _execute_web_search(self, query: str, num_results: int = 10) -> str:
        try:
            if not hasattr(self, "mcp_client"):
                await self._ensure_initialized()

            args = {"query": query, "num_results": num_results}
            search_response = await self.mcp_client.call_tool("web_search", args)

            if search_response.get("success") and search_response.get("results"):
                # Store results for later synthesis
                if not hasattr(self, "research_state"):
                    self.research_state = {
                        "search_results": [],
                        "key_findings": [],
                        "analyzed_sources": [],
                    }
                new_results = []
                for result_data in search_response["results"]:
                    search_result = SearchResult(
                        url=result_data.get("url", ""),
                        title=result_data.get("title", ""),
                        snippet=result_data.get("snippet", ""),
                        source_type="web_search",
                    )
                    new_results.append(search_result)
                    self.research_state["search_results"].append(search_result)

                results_summary = (
                    f"Found {len(new_results)} search results for '{query}':\n"
                )
                for i, result in enumerate(new_results, 1):
                    results_summary += f"{i}. {result.title}\n"
                    results_summary += f"   URL: {result.url}\n"
                    results_summary += f"   Snippet: {result.snippet}\n\n"

                return results_summary

            else:
                return f"Search failed: {search_response.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return f"Error performing web search: {str(e)}"

    async def _execute_webpage_analysis(
        self, url: str, extract_text: bool = True, summarize: bool = True
    ) -> str:
        try:
            if not hasattr(self, "mcp_client"):
                await self._ensure_initialized()

            args = {"url": url, "extract_text": extract_text, "summarize": summarize}
            analysis_response = await self.mcp_client.call_tool("analyze_webpage", args)

            if analysis_response.get("success"):
                content = analysis_response.get("content", "")
                summary = analysis_response.get("summary", "")
                title = analysis_response.get("title", "")
                word_count = analysis_response.get("word_count", 0)

                # Initialize research state if needed
                if not hasattr(self, "research_state"):
                    self.research_state = {
                        "search_results": [],
                        "analyzed_sources": [],
                        "key_findings": [],
                    }

                # Store analyzed source
                analysis_result = {
                    "url": url,
                    "title": title,
                    "content": content,
                    "summary": summary,
                    "word_count": word_count,
                }
                self.research_state["analyzed_sources"].append(analysis_result)

                # Extract key findings
                key_findings = await self._dspy_extract_key_findings(
                    content, getattr(self, "current_query", url)
                )
                self.research_state["key_findings"].extend(key_findings)

                result_summary = f"Analyzed webpage: {title}\n"
                result_summary += f"URL: {url}\n"
                result_summary += f"Word count: {word_count}\n"
                result_summary += f"Summary: {summary}\n"
                if key_findings:
                    result_summary += f"Key findings:\n"
                    for finding in key_findings:
                        result_summary += f"- {finding}\n"

                return result_summary
            else:
                return f"Analysis failed: {analysis_response.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"Error analyzing webpage: {e}")
            return f"Error analyzing webpage: {str(e)}"

    async def _dspy_extract_key_findings(self, content: str, query: str) -> List[str]:
        """Extract key findings using DSPy module."""
        try:
            self._ensure_openai_client()

            content_sample = content[:2000] if len(content) > 2000 else content

            findings_result = self.finding_module(
                content=content_sample, research_query=query
            )

            findings = [
                f.strip() for f in findings_result.key_findings.split("\n") if f.strip()
            ]
            return findings

        except Exception as e:
            logger.error(f"Error extracting key findings with DSPy: {e}")
            return []

    async def _dspy_synthesize_results(self) -> WebResearchResult:
        """Synthesize results using DSPy module."""
        try:
            self._ensure_openai_client()

            search_summary = []
            for result in self.research_state["search_results"]:
                search_summary.append(f"- {result.title}: {result.snippet}")

            search_results_text = (
                "\n".join(search_summary)
                if search_summary
                else "No search results found"
            )

            # Use DSPy synthesis module
            synthesis_result = self.synthesis_module(
                original_query=self.research_state["original_query"],
                search_results_summary=search_results_text,
                key_findings="\n".join(self.research_state["key_findings"]),
            )

            summary = synthesis_result.summary

            # Determine research depth (unchanged from original)
            research_depth = self._determine_research_depth()

            # Create final result object
            result = WebResearchResult(
                query=self.research_state["original_query"],
                search_results=self.research_state["search_results"],
                summary=summary,
                key_findings=self.research_state["key_findings"],
                sources_analyzed=len(self.research_state["analyzed_sources"]),
                research_depth=research_depth,
                react_trace=self.research_state["react_steps"],
                metadata={
                    "total_sources_found": len(self.research_state["search_results"]),
                    "react_cycles": len(self.research_state["react_steps"]),
                    "research_completed_at": datetime.now().isoformat(),
                    "methodology": "DSPy ReAct (Optimized Reasoning and Acting)",
                    "optimization_method": "DSPy_Automated",
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error synthesizing DSPy results: {e}")
            # Return a basic result even if synthesis fails
            return WebResearchResult(
                query=self.research_state["original_query"],
                search_results=self.research_state.get("search_results", []),
                summary=f"DSPy ReAct research completed with {len(self.research_state.get('analyzed_sources', []))} sources.",
                key_findings=self.research_state.get("key_findings", []),
                sources_analyzed=len(self.research_state.get("analyzed_sources", [])),
                research_depth="moderate",
                react_trace=self.research_state.get("react_steps", []),
                metadata={"error": str(e)},
            )

    def _determine_research_depth(self) -> str:
        cycles = len(self.research_state["react_steps"])
        sources_count = len(self.research_state["analyzed_sources"])
        findings_count = len(self.research_state["key_findings"])

        if cycles >= 4 and sources_count >= 5 and findings_count >= 10:
            return "DEEP"
        elif cycles >= 2 and sources_count >= 3 and findings_count >= 5:
            return "MODERATE"
        else:
            return "SURFACE"
