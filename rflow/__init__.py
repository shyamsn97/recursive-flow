"""recursive-flow (minimal): an LLM in a loop with a stateful REPL, recursive.

Quick start::

    from rflow import Flow
    from rflow.clients import OpenAIClient

    flow = Flow(OpenAIClient(model="gpt-4o"))
    print(flow.run("What is 17 * 23? Verify with code."))

The whole tree advances by synchronized steps; drive it yourself with
``start`` / ``step`` to inspect or visualize each tick::

    graph = flow.start("research X")
    while not graph.finished:
        graph = flow.step()
        ...  # inspect graph
"""

from rflow.clients import (
    AnthropicClient,
    LLMChannel,
    LLMClient,
    LLMUsage,
    OpenAIClient,
    TinkerClient,
    is_retryable,
    retry_transient,
)
from rflow.flow import Flow, ResumeError, find_code_blocks
from rflow.graph import (
    Action,
    ActionNode,
    CallLLM,
    ChildHandle,
    CodeObservation,
    DoneOutput,
    Edge,
    EdgesView,
    ErrorOutput,
    Exec,
    ExecAction,
    ExecOutput,
    Graph,
    LLMAction,
    LLMOutput,
    Node,
    NodesView,
    ObservationNode,
    Resume,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
    WaitRequest,
    inject,
    inject_output,
    is_action,
    is_code_observation,
    is_done,
    is_errored,
    is_exec_action,
    is_exec_output,
    is_llm_action,
    is_llm_output,
    is_observation,
    is_resume_action,
    is_resumed,
    is_supervising,
    is_user_query,
    parse_node_obj,
    prune_descendants_spawned_after,
    replace_last_action,
    replace_last_observation,
    replace_node,
    retrace_steps,
    truncate_after,
    truncate_agent,
)
from rflow.prompts import DEFAULT_BUILDER, SYSTEM_PROMPT, PromptBuilder
from rflow.repl import REPL, DoneSignal
from rflow.runtime import DockerRuntime, LocalRuntime, Runtime
from rflow.tools import FILE_TOOLS, get_tool_metadata, tool
from rflow.utils.trace import Trace, load_trace, save_trace

__all__ = [
    "Action",
    "ActionNode",
    "AnthropicClient",
    "CallLLM",
    "ChildHandle",
    "CodeObservation",
    "DEFAULT_BUILDER",
    "DockerRuntime",
    "DoneOutput",
    "DoneSignal",
    "Edge",
    "EdgesView",
    "ErrorOutput",
    "Exec",
    "ExecAction",
    "ExecOutput",
    "FILE_TOOLS",
    "Flow",
    "ResumeError",
    "Graph",
    "LLMAction",
    "LLMChannel",
    "LLMClient",
    "LLMOutput",
    "LLMUsage",
    "LocalRuntime",
    "Node",
    "NodesView",
    "ObservationNode",
    "OpenAIClient",
    "PromptBuilder",
    "REPL",
    "Resume",
    "ResumeAction",
    "Runtime",
    "SYSTEM_PROMPT",
    "SupervisingOutput",
    "TinkerClient",
    "Trace",
    "UserQuery",
    "WaitRequest",
    "find_code_blocks",
    "get_tool_metadata",
    "inject",
    "inject_output",
    "load_trace",
    "prune_descendants_spawned_after",
    "replace_last_action",
    "replace_last_observation",
    "replace_node",
    "retrace_steps",
    "save_trace",
    "truncate_after",
    "truncate_agent",
    "is_action",
    "is_code_observation",
    "is_done",
    "is_errored",
    "is_exec_action",
    "is_exec_output",
    "is_llm_action",
    "is_llm_output",
    "is_observation",
    "is_resume_action",
    "is_resumed",
    "is_retryable",
    "is_supervising",
    "is_user_query",
    "parse_node_obj",
    "retry_transient",
    "tool",
]
