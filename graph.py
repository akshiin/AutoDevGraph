from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agents import architect_node, developer_node, tester_node
from state import AgentState
from tools import all_tools


def _should_use_tools(state: AgentState) -> str:
    """Route developer output: if the agent made tool calls, execute them;
    otherwise it is done writing and we hand off to the tester."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "write_files"
    return "tester"


def _should_continue(state: AgentState) -> str:
    if state["error_log"] == "PASS" or state["iteration"] >= 3:
        return "end"
    return "developer"


def build_graph():
    tool_node = ToolNode(all_tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("architect", architect_node)
    workflow.add_node("developer", developer_node)
    workflow.add_node("write_files", tool_node)
    workflow.add_node("tester", tester_node)

    workflow.set_entry_point("architect")
    workflow.add_edge("architect", "developer")

    # ReAct loop: developer ↔ write_files until the agent stops making tool calls
    workflow.add_conditional_edges(
        "developer",
        _should_use_tools,
        {"write_files": "write_files", "tester": "tester"},
    )
    workflow.add_edge("write_files", "developer")

    workflow.add_conditional_edges(
        "tester",
        _should_continue,
        {"developer": "developer", "end": END},
    )

    return workflow.compile()
