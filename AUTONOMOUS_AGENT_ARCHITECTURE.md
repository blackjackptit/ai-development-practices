# Autonomous Agent Architecture

## Overview

Autonomous agents are AI systems that can perform tasks independently with minimal human intervention, making decisions, using tools, and adapting to changing conditions. This guide provides comprehensive architecture patterns for building autonomous agents using Claude CLI, LangChain, AutoGPT-style frameworks, or custom implementations.

**Core Concept:** An autonomous agent perceives its environment, reasons about goals, makes decisions, takes actions using tools, and learns from outcomes in a continuous loop.

```
┌────────────────────────────────────────────────────────────────┐
│                   AUTONOMOUS AGENT LOOP                        │
│                                                                │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌───────┐│
│  │ Perceive │────▶│  Reason  │────▶│  Decide  │────▶│  Act  ││
│  │ (Observe)│     │  (Plan)  │     │ (Choose) │     │(Tools)││
│  └────┬─────┘     └──────────┘     └──────────┘     └───┬───┘│
│       │                                                    │   │
│       │          ┌─────────────┐                          │   │
│       └──────────│   Memory    │◀─────────────────────────┘   │
│                  │  (Context)  │                              │
│                  └─────────────┘                              │
└────────────────────────────────────────────────────────────────┘
```

**Key Capabilities:**
- Goal-oriented behavior (achieves specific objectives)
- Tool use (file system, APIs, databases, web search)
- Multi-step reasoning (breaks down complex tasks)
- Memory management (short-term and long-term)
- Self-reflection and error correction
- Cost-aware execution

---

## Table of Contents

1. [Core Components](#1-core-components)
2. [Agent Loop Architecture](#2-agent-loop-architecture)
3. [Tool System Design](#3-tool-system-design)
4. [Memory Architecture](#4-memory-architecture)
5. [Planning and Reasoning](#5-planning-and-reasoning)
6. [Claude CLI Agent Implementation](#6-claude-cli-agent-implementation)
7. [Complete Agent Examples](#7-complete-agent-examples)
8. [Multi-Agent Systems](#8-multi-agent-systems)
9. [Monitoring and Control](#9-monitoring-and-control)
10. [Best Practices](#10-best-practices)

---

## 1. Core Components

### 1.1 Agent Architecture Stack

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                   (CLI, Web, API, Chat)                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                      AGENT CONTROLLER                            │
│         • Task decomposition  • Goal management                  │
│         • Cost tracking       • Safety checks                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                         AGENT LOOP                               │
│                                                                  │
│  ┌───────────────┐    ┌────────────────┐    ┌────────────────┐ │
│  │  Perception   │───▶│   Reasoning    │───▶│     Action     │ │
│  │  • Observe    │    │  • Plan steps  │    │  • Use tools   │ │
│  │  • Parse      │    │  • Evaluate    │    │  • Execute     │ │
│  └───────────────┘    └────────────────┘    └────────────────┘ │
│         │                      │                      │          │
│         └──────────────────────┼──────────────────────┘          │
│                                │                                 │
└────────────────────────────────┼─────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────┐
│                          MEMORY SYSTEM                           │
│  • Working memory (conversation)  • Long-term memory (vector DB) │
│  • Episodic memory (past actions) • Semantic memory (knowledge)  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                          TOOL LAYER                              │
│  • File system  • Web search  • Code execution  • APIs           │
│  • Database     • Shell       • Calculator      • Custom tools   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Essential Agent Components

```python
# entities/agent.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class AgentState(Enum):
    """Agent execution state."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentGoal:
    """Represents an agent's goal."""
    id: str
    description: str
    success_criteria: List[str]
    max_steps: int = 50
    max_cost: float = 1.0  # USD
    priority: int = 1

@dataclass
class AgentAction:
    """Represents an action taken by the agent."""
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str
    timestamp: float

@dataclass
class AgentObservation:
    """Represents an observation from environment."""
    content: str
    source: str  # "tool", "user", "system"
    timestamp: float
    success: bool = True
    error: Optional[str] = None

@dataclass
class AgentMemory:
    """Agent's memory system."""
    working_memory: List[str] = field(default_factory=list)  # Recent context
    episodic_memory: List[AgentAction] = field(default_factory=list)  # Past actions
    semantic_memory: Dict[str, Any] = field(default_factory=dict)  # Knowledge

    def add_to_working_memory(self, content: str, max_items: int = 10):
        """Add to working memory with limit."""
        self.working_memory.append(content)
        if len(self.working_memory) > max_items:
            self.working_memory = self.working_memory[-max_items:]

@dataclass
class Agent:
    """Autonomous agent."""
    id: str
    name: str
    goal: AgentGoal
    state: AgentState = AgentState.IDLE
    memory: AgentMemory = field(default_factory=AgentMemory)
    steps_taken: int = 0
    total_cost: float = 0.0
    observations: List[AgentObservation] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)

    def can_continue(self) -> bool:
        """Check if agent can continue execution."""
        return (
            self.steps_taken < self.goal.max_steps and
            self.total_cost < self.goal.max_cost and
            self.state not in [AgentState.COMPLETED, AgentState.FAILED]
        )

    def add_observation(self, observation: AgentObservation):
        """Add observation and update working memory."""
        self.observations.append(observation)
        self.memory.add_to_working_memory(
            f"Observation: {observation.content}"
        )

    def add_action(self, action: AgentAction):
        """Add action to history."""
        self.actions.append(action)
        self.memory.episodic_memory.append(action)
        self.steps_taken += 1
```

---

## 2. Agent Loop Architecture

### 2.1 ReAct Pattern (Reasoning + Acting)

The ReAct pattern interleaves reasoning and acting:

```
Thought: I need to find the weather in San Francisco
Action: web_search("San Francisco weather")
Observation: Temperature is 65°F, sunny

Thought: Now I have the weather information
Action: final_answer("The weather in San Francisco is 65°F and sunny")
```

**Implementation:**

```python
# use_cases/agent_loop.py
from typing import List, Optional, Tuple
from entities.agent import Agent, AgentAction, AgentObservation, AgentState
from use_cases.interfaces import LLMGatewayInterface, ToolRegistryInterface

class AgentLoopUseCase:
    """
    Implements the core agent loop: Think → Act → Observe → Repeat
    Based on ReAct pattern (Reasoning + Acting)
    """

    # System prompt for ReAct agent
    SYSTEM_PROMPT = """You are an autonomous agent that can use tools to accomplish tasks.

Follow this pattern:
Thought: [your reasoning about what to do next]
Action: [tool_name] [parameters as JSON]
Observation: [result of the action - this will be provided by the system]

Continue this loop until you have completed the task, then:
Thought: [your final reasoning]
Action: final_answer [your response to the user]

Available tools:
{tool_descriptions}

Important rules:
1. Always start with "Thought:" to explain your reasoning
2. Then use "Action:" to call a tool
3. Wait for "Observation:" before continuing
4. Use "Action: final_answer" when task is complete
5. Be cost-conscious - use tools efficiently
6. If you encounter errors, try alternative approaches

Goal: {goal}
"""

    def __init__(
        self,
        llm_gateway: LLMGatewayInterface,
        tool_registry: ToolRegistryInterface
    ):
        self.llm = llm_gateway
        self.tools = tool_registry

    def execute(self, agent: Agent) -> Agent:
        """
        Execute the agent loop until goal is achieved or limits reached.
        """

        # Build system prompt with available tools
        tool_descriptions = self._format_tool_descriptions()
        system_prompt = self.SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            goal=agent.goal.description
        )

        # Agent loop
        while agent.can_continue():
            # 1. THINK: Get agent's next thought and action
            agent.state = AgentState.THINKING
            thought, action = self._think(agent, system_prompt)

            # Check if agent wants to finish
            if action.tool_name == "final_answer":
                agent.state = AgentState.COMPLETED
                return agent

            # 2. ACT: Execute the action
            agent.state = AgentState.ACTING
            observation = self._act(action)

            # 3. OBSERVE: Record observation
            agent.add_observation(observation)

            # 4. REFLECT: Check if should continue
            agent.state = AgentState.REFLECTING
            if observation.error:
                # Agent failed - could implement retry logic here
                if agent.steps_taken >= agent.goal.max_steps:
                    agent.state = AgentState.FAILED
                    return agent

        # Reached limits
        agent.state = AgentState.FAILED
        return agent

    def _think(
        self,
        agent: Agent,
        system_prompt: str
    ) -> Tuple[str, AgentAction]:
        """
        Agent thinks about next action.
        Returns: (thought, action)
        """

        # Build context from working memory
        context = "\n".join(agent.memory.working_memory)

        # Generate next thought and action
        prompt = f"""{system_prompt}

Previous context:
{context}

What is your next thought and action?"""

        response = self.llm.generate(
            prompt=prompt,
            model="claude-3-sonnet-20240229",  # Sonnet for reasoning
            max_tokens=500
        )

        # Parse response
        thought, action = self._parse_react_response(response.text)

        # Record action
        agent.add_action(action)
        agent.total_cost += response.cost

        return thought, action

    def _parse_react_response(self, text: str) -> Tuple[str, AgentAction]:
        """Parse Thought and Action from LLM response."""
        import re
        import json
        from datetime import datetime

        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Extract action
        action_match = re.search(r"Action:\s*(\w+)\s*(.+?)(?=Observation:|$)", text, re.DOTALL)

        if not action_match:
            # Default to final_answer if no action found
            return thought, AgentAction(
                tool_name="final_answer",
                parameters={"answer": text},
                reasoning=thought,
                timestamp=datetime.utcnow().timestamp()
            )

        tool_name = action_match.group(1).strip()
        params_str = action_match.group(2).strip()

        # Try to parse parameters as JSON
        try:
            if params_str.startswith("{"):
                parameters = json.loads(params_str)
            else:
                # Simple string parameter
                parameters = {"input": params_str}
        except json.JSONDecodeError:
            parameters = {"input": params_str}

        return thought, AgentAction(
            tool_name=tool_name,
            parameters=parameters,
            reasoning=thought,
            timestamp=datetime.utcnow().timestamp()
        )

    def _act(self, action: AgentAction) -> AgentObservation:
        """Execute action using tool registry."""
        from datetime import datetime

        try:
            # Get tool and execute
            tool = self.tools.get_tool(action.tool_name)

            if not tool:
                return AgentObservation(
                    content="",
                    source="system",
                    timestamp=datetime.utcnow().timestamp(),
                    success=False,
                    error=f"Tool '{action.tool_name}' not found"
                )

            result = tool.execute(**action.parameters)

            return AgentObservation(
                content=str(result),
                source="tool",
                timestamp=datetime.utcnow().timestamp(),
                success=True
            )

        except Exception as e:
            return AgentObservation(
                content="",
                source="tool",
                timestamp=datetime.utcnow().timestamp(),
                success=False,
                error=str(e)
            )

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for prompt."""
        tools = self.tools.list_tools()
        descriptions = []

        for tool in tools:
            descriptions.append(
                f"- {tool.name}: {tool.description}\n"
                f"  Parameters: {tool.parameters}"
            )

        return "\n".join(descriptions)
```

### 2.2 Plan-and-Execute Pattern

Alternative to ReAct: plan all steps upfront, then execute.

```python
# use_cases/plan_and_execute_agent.py
from typing import List
from entities.agent import Agent, AgentAction

class PlanAndExecuteAgentUseCase:
    """
    Agent that plans all steps first, then executes them.
    Better for tasks with clear structure.
    """

    PLANNING_PROMPT = """You are an AI agent that creates execution plans.

Given this goal: {goal}

Create a step-by-step plan using available tools:
{tool_descriptions}

Output plan as JSON array of steps:
[
  {{"step": 1, "tool": "tool_name", "params": {{"key": "value"}}, "reasoning": "why this step"}},
  ...
]

Plan:"""

    def __init__(
        self,
        llm_gateway: LLMGatewayInterface,
        tool_registry: ToolRegistryInterface
    ):
        self.llm = llm_gateway
        self.tools = tool_registry

    def execute(self, agent: Agent) -> Agent:
        """Execute plan-and-execute loop."""

        # Phase 1: PLAN
        plan = self._create_plan(agent)

        # Phase 2: EXECUTE
        for step in plan:
            if not agent.can_continue():
                break

            action = AgentAction(
                tool_name=step["tool"],
                parameters=step["params"],
                reasoning=step["reasoning"],
                timestamp=datetime.utcnow().timestamp()
            )

            # Execute step
            observation = self._execute_step(action)
            agent.add_action(action)
            agent.add_observation(observation)

            # Check for errors
            if observation.error:
                # Replan if error
                plan = self._replan(agent, plan, step)

        agent.state = AgentState.COMPLETED
        return agent

    def _create_plan(self, agent: Agent) -> List[dict]:
        """Create execution plan."""
        import json

        tool_descriptions = self._format_tool_descriptions()
        prompt = self.PLANNING_PROMPT.format(
            goal=agent.goal.description,
            tool_descriptions=tool_descriptions
        )

        response = self.llm.generate(prompt, "claude-3-sonnet-20240229", 1000)
        agent.total_cost += response.cost

        # Parse plan
        try:
            plan = json.loads(response.text)
            return plan
        except json.JSONDecodeError:
            # Fallback: extract JSON from markdown code block
            import re
            json_match = re.search(r"```json\n(.+?)\n```", response.text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(1))
                return plan
            raise ValueError("Failed to parse plan")
```

---

## 3. Tool System Design

### 3.1 Tool Interface

```python
# use_cases/interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass

@dataclass
class ToolParameter:
    """Parameter definition for a tool."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None

@dataclass
class ToolDefinition:
    """Metadata about a tool."""
    name: str
    description: str
    parameters: List[ToolParameter]
    cost_estimate: float = 0.0  # Estimated cost in USD

class ToolInterface(ABC):
    """Interface for agent tools."""

    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Get tool metadata."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool."""
        pass

class ToolRegistryInterface(ABC):
    """Interface for tool registry."""

    @abstractmethod
    def register_tool(self, tool: ToolInterface) -> None:
        """Register a tool."""
        pass

    @abstractmethod
    def get_tool(self, name: str) -> ToolInterface:
        """Get tool by name."""
        pass

    @abstractmethod
    def list_tools(self) -> List[ToolDefinition]:
        """List all available tools."""
        pass
```

### 3.2 Built-in Tools

```python
# adapters/tools/file_system_tool.py
import os
from pathlib import Path
from use_cases.interfaces import ToolInterface, ToolDefinition, ToolParameter

class ReadFileTool(ToolInterface):
    """Tool to read files from filesystem."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read contents of a file",
            parameters=[
                ToolParameter("file_path", "string", "Path to file to read"),
                ToolParameter("max_lines", "number", "Maximum lines to read", required=False, default=1000)
            ],
            cost_estimate=0.0
        )

    def execute(self, file_path: str, max_lines: int = 1000) -> str:
        """Read file contents."""
        try:
            path = Path(file_path).resolve()

            # Safety: Only allow reading from current directory tree
            cwd = Path.cwd().resolve()
            if not str(path).startswith(str(cwd)):
                raise ValueError(f"Access denied: {file_path}")

            with open(path, 'r') as f:
                lines = f.readlines()[:max_lines]

            return "".join(lines)

        except Exception as e:
            return f"Error reading file: {e}"

class WriteFileTool(ToolInterface):
    """Tool to write files to filesystem."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description="Write content to a file",
            parameters=[
                ToolParameter("file_path", "string", "Path to file to write"),
                ToolParameter("content", "string", "Content to write")
            ],
            cost_estimate=0.0
        )

    def execute(self, file_path: str, content: str) -> str:
        """Write file contents."""
        try:
            path = Path(file_path).resolve()

            # Safety check
            cwd = Path.cwd().resolve()
            if not str(path).startswith(str(cwd)):
                raise ValueError(f"Access denied: {file_path}")

            with open(path, 'w') as f:
                f.write(content)

            return f"Successfully wrote {len(content)} characters to {file_path}"

        except Exception as e:
            return f"Error writing file: {e}"

# adapters/tools/web_search_tool.py
import requests
from use_cases.interfaces import ToolInterface, ToolDefinition, ToolParameter

class WebSearchTool(ToolInterface):
    """Tool to search the web."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for information",
            parameters=[
                ToolParameter("query", "string", "Search query"),
                ToolParameter("num_results", "number", "Number of results", required=False, default=5)
            ],
            cost_estimate=0.001  # Cost of API call
        )

    def execute(self, query: str, num_results: int = 5) -> str:
        """Search web using API."""
        try:
            # Example using DuckDuckGo (no API key needed)
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json"}

            response = requests.get(url, params=params)
            data = response.json()

            # Format results
            results = []
            for item in data.get("RelatedTopics", [])[:num_results]:
                if "Text" in item:
                    results.append(item["Text"])

            return "\n\n".join(results) if results else "No results found"

        except Exception as e:
            return f"Error searching web: {e}"

# adapters/tools/python_executor_tool.py
import sys
from io import StringIO
from use_cases.interfaces import ToolInterface, ToolDefinition, ToolParameter

class PythonExecutorTool(ToolInterface):
    """Tool to execute Python code."""

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="python_exec",
            description="Execute Python code and return output",
            parameters=[
                ToolParameter("code", "string", "Python code to execute")
            ],
            cost_estimate=0.0
        )

    def execute(self, code: str) -> str:
        """Execute Python code safely."""
        try:
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            # Execute code with restricted globals
            allowed_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    # Add more as needed
                }
            }

            exec(code, allowed_globals)

            # Get output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            return output if output else "Code executed successfully (no output)"

        except Exception as e:
            sys.stdout = old_stdout
            return f"Error executing code: {e}"
```

### 3.3 Tool Registry

```python
# adapters/tool_registry.py
from typing import Dict, List
from use_cases.interfaces import (
    ToolInterface,
    ToolRegistryInterface,
    ToolDefinition
)

class ToolRegistry(ToolRegistryInterface):
    """Registry for agent tools."""

    def __init__(self):
        self.tools: Dict[str, ToolInterface] = {}

    def register_tool(self, tool: ToolInterface) -> None:
        """Register a tool."""
        definition = tool.get_definition()
        self.tools[definition.name] = tool

    def get_tool(self, name: str) -> ToolInterface:
        """Get tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """List all tools."""
        return [tool.get_definition() for tool in self.tools.values()]

# Usage
def create_tool_registry() -> ToolRegistry:
    """Create and populate tool registry."""
    registry = ToolRegistry()

    # Register built-in tools
    registry.register_tool(ReadFileTool())
    registry.register_tool(WriteFileTool())
    registry.register_tool(WebSearchTool(api_key=os.getenv("SEARCH_API_KEY")))
    registry.register_tool(PythonExecutorTool())

    # Register custom tools
    # registry.register_tool(CustomTool())

    return registry
```

---

## 4. Memory Architecture

### 4.1 Multi-Level Memory System

```python
# adapters/memory/agent_memory_system.py
from typing import List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MemoryEntry:
    """Entry in agent's memory."""
    content: str
    timestamp: datetime
    importance: float  # 0-1
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

class AgentMemorySystem:
    """
    Multi-level memory system for agents.

    Levels:
    1. Working Memory: Recent context (limited capacity)
    2. Episodic Memory: Past experiences and actions
    3. Semantic Memory: General knowledge
    4. Long-term Memory: Vector database for retrieval
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        working_memory_size: int = 10,
        episodic_memory_size: int = 100
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

        # Different memory levels
        self.working_memory: List[MemoryEntry] = []
        self.episodic_memory: List[MemoryEntry] = []
        self.semantic_memory: Dict[str, Any] = {}

        # Capacity limits
        self.working_memory_size = working_memory_size
        self.episodic_memory_size = episodic_memory_size

    def add_to_working_memory(self, content: str, importance: float = 0.5):
        """Add to working memory with recency bias."""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.utcnow(),
            importance=importance
        )

        self.working_memory.append(entry)

        # Prune if exceeds capacity
        if len(self.working_memory) > self.working_memory_size:
            # Keep most important and most recent
            self.working_memory = sorted(
                self.working_memory,
                key=lambda x: x.importance * 0.5 + (1.0 if x.timestamp > datetime.utcnow() - timedelta(minutes=5) else 0.0),
                reverse=True
            )[:self.working_memory_size]

    def add_to_episodic_memory(self, content: str, importance: float = 0.5):
        """Add to episodic memory (experiences and actions)."""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.utcnow(),
            importance=importance,
            embedding=self.embedding_model.encode(content)
        )

        self.episodic_memory.append(entry)

        # Store in long-term vector database
        self.vector_store.add(entry)

        # Prune old episodic memories
        if len(self.episodic_memory) > self.episodic_memory_size:
            self.episodic_memory = self.episodic_memory[-self.episodic_memory_size:]

    def retrieve_relevant_memories(
        self,
        query: str,
        k: int = 5,
        min_importance: float = 0.3
    ) -> List[MemoryEntry]:
        """Retrieve relevant memories using semantic search."""
        query_embedding = self.embedding_model.encode(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)

        # Filter by importance
        relevant = [
            entry for entry in results
            if entry.importance >= min_importance
        ]

        return relevant

    def consolidate_memories(self):
        """
        Consolidate memories (move from episodic to semantic).
        Runs periodically to extract general knowledge.
        """
        # Use LLM to extract patterns and knowledge from episodic memory
        # This is a simplified version
        recent_episodes = self.episodic_memory[-20:]

        # Extract knowledge (simplified - would use LLM in practice)
        for episode in recent_episodes:
            if "learned:" in episode.content.lower():
                knowledge = episode.content.split("learned:")[1].strip()
                self.semantic_memory[knowledge] = episode.timestamp

    def get_context(self) -> str:
        """Get formatted context from all memory levels."""
        context_parts = []

        # Working memory (most recent)
        if self.working_memory:
            recent = "\n".join([m.content for m in self.working_memory[-5:]])
            context_parts.append(f"Recent context:\n{recent}")

        # Semantic memory (knowledge)
        if self.semantic_memory:
            knowledge = "\n".join([f"- {k}" for k in list(self.semantic_memory.keys())[:5]])
            context_parts.append(f"\nKnown facts:\n{knowledge}")

        return "\n\n".join(context_parts)
```

---

## 5. Planning and Reasoning

### 5.1 Hierarchical Task Decomposition

```python
# use_cases/hierarchical_planner.py
from typing import List
from dataclasses import dataclass

@dataclass
class Task:
    """Represents a task in the hierarchy."""
    id: str
    description: str
    subtasks: List['Task'] = None
    completed: bool = False
    parent_id: str = None

class HierarchicalPlannerUseCase:
    """
    Breaks down complex goals into hierarchical tasks.
    Uses LLM to recursively decompose tasks.
    """

    DECOMPOSITION_PROMPT = """Break down this task into 3-5 subtasks:

Task: {task}

Subtasks (as JSON array):
[
  "subtask 1 description",
  "subtask 2 description",
  ...
]

If task is simple and cannot be decomposed further, return empty array: []

Subtasks:"""

    def __init__(self, llm_gateway: LLMGatewayInterface):
        self.llm = llm_gateway

    def decompose_task(self, task: Task, max_depth: int = 3, current_depth: int = 0) -> Task:
        """Recursively decompose task into subtasks."""
        import json

        if current_depth >= max_depth:
            return task

        # Ask LLM to decompose
        prompt = self.DECOMPOSITION_PROMPT.format(task=task.description)
        response = self.llm.generate(prompt, "claude-3-sonnet-20240229", 500)

        try:
            subtask_descriptions = json.loads(response.text)

            if not subtask_descriptions:
                # Atomic task
                return task

            # Create subtasks
            task.subtasks = []
            for i, desc in enumerate(subtask_descriptions):
                subtask = Task(
                    id=f"{task.id}.{i+1}",
                    description=desc,
                    parent_id=task.id
                )

                # Recursively decompose
                subtask = self.decompose_task(subtask, max_depth, current_depth + 1)
                task.subtasks.append(subtask)

            return task

        except json.JSONDecodeError:
            # Failed to parse - treat as atomic
            return task

    def get_next_atomic_task(self, root_task: Task) -> Optional[Task]:
        """Get next incomplete atomic task (depth-first)."""
        if not root_task.subtasks:
            # Atomic task
            return root_task if not root_task.completed else None

        # Has subtasks - check them
        for subtask in root_task.subtasks:
            next_task = self.get_next_atomic_task(subtask)
            if next_task:
                return next_task

        return None
```

---

## 6. Claude CLI Agent Implementation

### 6.1 Claude CLI Tool Use

Claude CLI supports tool use natively. Here's how to build an agent:

```python
# claude_cli_agent.py
import anthropic
import json
from typing import List, Dict, Any, Callable

class ClaudeAgentCLI:
    """
    Autonomous agent using Claude CLI with native tool use.
    Uses Claude's built-in tool calling capabilities.
    """

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.tools: Dict[str, Callable] = {}
        self.conversation_history = []

    def register_tool(self, name: str, function: Callable, schema: Dict):
        """Register a tool with its implementation and schema."""
        self.tools[name] = function

        # Store schema for Claude
        if not hasattr(self, 'tool_schemas'):
            self.tool_schemas = []
        self.tool_schemas.append(schema)

    def run(self, goal: str, max_iterations: int = 10) -> str:
        """
        Run agent to achieve goal.

        Uses Claude's native tool calling via the API.
        """

        # Initialize conversation
        messages = [{
            "role": "user",
            "content": goal
        }]

        for iteration in range(max_iterations):
            # Call Claude with tools
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                tools=self.tool_schemas,
                messages=messages
            )

            # Check if Claude wants to use a tool
            if response.stop_reason == "tool_use":
                # Execute tool calls
                tool_results = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input

                        print(f"\n[Agent] Using tool: {tool_name}")
                        print(f"[Agent] Input: {json.dumps(tool_input, indent=2)}")

                        # Execute tool
                        if tool_name in self.tools:
                            try:
                                result = self.tools[tool_name](**tool_input)
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": content_block.id,
                                    "content": str(result)
                                })
                                print(f"[Agent] Result: {result}")
                            except Exception as e:
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": content_block.id,
                                    "content": f"Error: {str(e)}",
                                    "is_error": True
                                })
                                print(f"[Agent] Error: {e}")

                # Add assistant message and tool results to conversation
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

            elif response.stop_reason == "end_turn":
                # Claude is done
                final_response = ""
                for content_block in response.content:
                    if hasattr(content_block, "text"):
                        final_response += content_block.text

                print(f"\n[Agent] Final answer: {final_response}")
                return final_response

            else:
                # Unexpected stop reason
                print(f"\n[Agent] Stopped: {response.stop_reason}")
                break

        return "Agent reached maximum iterations"

# Example usage
def main():
    agent = ClaudeAgentCLI(api_key="your-api-key")

    # Define tools
    def read_file(file_path: str) -> str:
        """Read a file from the filesystem."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    def write_file(file_path: str, content: str) -> str:
        """Write content to a file."""
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error: {e}"

    def web_search(query: str) -> str:
        """Search the web."""
        # Simplified - would use real search API
        return f"Search results for: {query}\n1. Result 1\n2. Result 2"

    # Register tools with schemas
    agent.register_tool(
        "read_file",
        read_file,
        {
            "name": "read_file",
            "description": "Read the contents of a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["file_path"]
            }
        }
    )

    agent.register_tool(
        "write_file",
        write_file,
        {
            "name": "write_file",
            "description": "Write content to a file",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    )

    agent.register_tool(
        "web_search",
        web_search,
        {
            "name": "web_search",
            "description": "Search the web for information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    )

    # Run agent
    result = agent.run(
        goal="Research Python best practices and create a summary document called python_best_practices.md"
    )

    print(f"\n=== Agent completed ===\n{result}")

if __name__ == "__main__":
    main()
```

---

## 7. Complete Agent Examples

### 7.1 Code Analysis Agent

```python
# examples/code_analyzer_agent.py
"""
Autonomous agent that analyzes code quality.
"""

def create_code_analyzer_agent():
    """Create agent that analyzes code."""

    agent = ClaudeAgentCLI(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Register tools
    agent.register_tool("read_file", read_file_tool, READ_FILE_SCHEMA)
    agent.register_tool("list_files", list_files_tool, LIST_FILES_SCHEMA)
    agent.register_tool("python_exec", python_exec_tool, PYTHON_EXEC_SCHEMA)

    # Run analysis
    result = agent.run("""
    Analyze the Python code in the current directory:
    1. List all .py files
    2. Read each file
    3. Check for:
       - Code complexity
       - Missing docstrings
       - Potential bugs
       - Security issues
    4. Create a report in code_analysis_report.md
    """)

    return result
```

### 7.2 Data Processing Agent

```python
# examples/data_processor_agent.py
"""
Agent that processes data files.
"""

def create_data_processor_agent():
    """Create agent that processes data."""

    agent = ClaudeAgentCLI(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Register tools
    agent.register_tool("read_csv", read_csv_tool, READ_CSV_SCHEMA)
    agent.register_tool("python_exec", python_exec_tool, PYTHON_EXEC_SCHEMA)
    agent.register_tool("write_file", write_file_tool, WRITE_FILE_SCHEMA)

    # Run processing
    result = agent.run("""
    Process the sales_data.csv file:
    1. Load the data
    2. Calculate summary statistics
    3. Identify top 10 customers by revenue
    4. Create visualizations
    5. Generate a report with insights
    """)

    return result
```

---

## 8. Multi-Agent Systems

### 8.1 Coordinator-Worker Pattern

```python
# use_cases/multi_agent_system.py
from typing import List, Dict
from entities.agent import Agent, AgentGoal

class MultiAgentCoordinator:
    """
    Coordinates multiple specialized agents.

    Pattern: One coordinator delegates to multiple worker agents.
    """

    def __init__(
        self,
        coordinator_llm: LLMGatewayInterface,
        worker_agents: Dict[str, Agent]
    ):
        self.coordinator = coordinator_llm
        self.workers = worker_agents

    def execute(self, goal: str) -> str:
        """
        Coordinate agents to achieve goal.

        1. Coordinator breaks down task
        2. Assigns subtasks to specialized workers
        3. Collects results
        4. Synthesizes final answer
        """

        # Phase 1: Decompose task
        subtasks = self._decompose_task(goal)

        # Phase 2: Assign to workers
        results = {}
        for subtask in subtasks:
            worker_name = self._select_worker(subtask)
            worker = self.workers[worker_name]

            # Execute subtask
            result = self._execute_subtask(worker, subtask)
            results[subtask] = result

        # Phase 3: Synthesize results
        final_answer = self._synthesize_results(goal, results)

        return final_answer

    def _decompose_task(self, goal: str) -> List[str]:
        """Decompose task into subtasks."""
        prompt = f"""Break down this goal into specific subtasks:

Goal: {goal}

Available workers:
{self._format_worker_capabilities()}

Subtasks (one per line):"""

        response = self.coordinator.generate(prompt, "claude-3-sonnet-20240229", 500)

        # Parse subtasks
        subtasks = [line.strip() for line in response.text.split("\n") if line.strip()]
        return subtasks

    def _select_worker(self, subtask: str) -> str:
        """Select best worker for subtask."""
        # Simple keyword matching - could use LLM for better selection
        for worker_name, worker in self.workers.items():
            if any(keyword in subtask.lower() for keyword in worker.goal.description.lower().split()):
                return worker_name

        # Default to first worker
        return list(self.workers.keys())[0]

# Example usage
def create_multi_agent_system():
    """Create multi-agent system with specialized workers."""

    # Create specialized agents
    researcher = Agent(
        id="researcher",
        name="Research Agent",
        goal=AgentGoal(
            id="research",
            description="Search web and gather information",
            success_criteria=["Found relevant information"],
            max_steps=20
        )
    )

    coder = Agent(
        id="coder",
        name="Coding Agent",
        goal=AgentGoal(
            id="code",
            description="Write and execute code",
            success_criteria=["Code works correctly"],
            max_steps=30
        )
    )

    writer = Agent(
        id="writer",
        name="Writing Agent",
        goal=AgentGoal(
            id="write",
            description="Create documents and reports",
            success_criteria=["Document created"],
            max_steps=15
        )
    )

    # Create coordinator
    coordinator = MultiAgentCoordinator(
        coordinator_llm=anthropic_gateway,
        worker_agents={
            "researcher": researcher,
            "coder": coder,
            "writer": writer
        }
    )

    # Execute complex task
    result = coordinator.execute(
        "Research Python async programming, write example code, and create a tutorial document"
    )

    return result
```

---

## 9. Monitoring and Control

### 9.1 Agent Dashboard

```python
# adapters/monitoring/agent_dashboard.py
from typing import List
from entities.agent import Agent, AgentState
from dataclasses import asdict
import json

class AgentDashboard:
    """Monitor and control running agents."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def register_agent(self, agent: Agent):
        """Register agent for monitoring."""
        self.agents[agent.id] = agent

    def get_status(self, agent_id: str) -> Dict:
        """Get agent status."""
        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": "Agent not found"}

        return {
            "id": agent.id,
            "name": agent.name,
            "state": agent.state.value,
            "steps_taken": agent.steps_taken,
            "total_cost": agent.total_cost,
            "progress": agent.steps_taken / agent.goal.max_steps,
            "can_continue": agent.can_continue()
        }

    def get_all_agents(self) -> List[Dict]:
        """Get status of all agents."""
        return [self.get_status(agent_id) for agent_id in self.agents.keys()]

    def pause_agent(self, agent_id: str):
        """Pause agent execution."""
        agent = self.agents.get(agent_id)
        if agent and agent.state not in [AgentState.COMPLETED, AgentState.FAILED]:
            agent.state = AgentState.IDLE

    def resume_agent(self, agent_id: str):
        """Resume paused agent."""
        agent = self.agents.get(agent_id)
        if agent and agent.state == AgentState.IDLE:
            agent.state = AgentState.THINKING

    def stop_agent(self, agent_id: str):
        """Stop agent execution."""
        agent = self.agents.get(agent_id)
        if agent:
            agent.state = AgentState.FAILED

    def export_agent_history(self, agent_id: str) -> str:
        """Export agent's action history."""
        agent = self.agents.get(agent_id)
        if not agent:
            return "{}"

        history = {
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "goal": agent.goal.description
            },
            "actions": [asdict(action) for action in agent.actions],
            "observations": [asdict(obs) for obs in agent.observations],
            "final_state": agent.state.value,
            "total_cost": agent.total_cost,
            "steps_taken": agent.steps_taken
        }

        return json.dumps(history, indent=2, default=str)
```

---

## 10. Best Practices

### 10.1 Dos

✅ **DO** set clear success criteria for goals
✅ **DO** limit max steps and cost to prevent runaway agents
✅ **DO** implement tool safety checks (filesystem access, API calls)
✅ **DO** log all actions for debugging and auditing
✅ **DO** use ReAct pattern for transparent reasoning
✅ **DO** implement error handling and recovery
✅ **DO** track costs continuously
✅ **DO** use appropriate models (Sonnet for reasoning, Haiku for simple tasks)
✅ **DO** implement human-in-the-loop for critical decisions
✅ **DO** test agents in sandbox environments first

### 10.2 Don'ts

❌ **DON'T** allow unlimited tool access
❌ **DON'T** run agents without cost limits
❌ **DON'T** skip safety validation for tools
❌ **DON'T** let agents modify critical files without confirmation
❌ **DON'T** use agents for tasks better suited for traditional code
❌ **DON'T** ignore failed actions (implement retry logic)
❌ **DON'T** forget to clean up resources after agent completes
❌ **DON'T** run multiple agents concurrently without coordination
❌ **DON'T** expose sensitive data to agent logs
❌ **DON'T** deploy to production without extensive testing

### 10.3 Safety Checklist

- [ ] Max steps limit configured
- [ ] Max cost limit configured
- [ ] Tool access restricted (whitelist approach)
- [ ] File system access sandboxed
- [ ] API rate limits enforced
- [ ] Sensitive data redacted from logs
- [ ] Human approval required for destructive actions
- [ ] Agent can be paused/stopped
- [ ] All actions logged
- [ ] Error handling implemented
- [ ] Timeout configured
- [ ] Resource cleanup on completion
- [ ] Testing in isolated environment

---

## References

### Research Papers
- **"ReAct: Synergizing Reasoning and Acting in Language Models"** (2022)
- **"Toolformer: Language Models Can Teach Themselves to Use Tools"** (2023)
- **"Voyager: An Open-Ended Embodied Agent with Large Language Models"** (2023)

### Related Architecture Documents
- [Clean Architecture](CLEAN_ARCHITECTURE.md) - Code organization for agents
- [System Architecture](SYSTEM_ARCHITECTURE.md) - System design
- [Security Architecture](SECURITY_ARCHITECTURE.md) - Securing agent systems
- [Cost-Efficient Architecture](COST_EFFICIENT_ARCHITECTURE.md) - Cost management

### External Resources
- [Claude API Tool Use Documentation](https://docs.anthropic.com/claude/docs/tool-use)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

---

**Version:** 1.0
**Last Updated:** February 6, 2026
**Total:** 2,500+ lines of autonomous agent architecture guidance
**Status:** Active
