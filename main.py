import os

from typing import List, TypedDict, Annotated
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

# --- 1. TOOLS: The Agent's "Hands" ---


@tool
def write_to_disk(filename: str, content: str, folder: str = "backend"):
    """Writes code content to a specific file in the local filesystem."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote {filename} to {folder}/"


tools = [write_to_disk]
tool_node = ToolNode(tools)

# --- 2. STATE & SCHEMAS ---


class TechnicalSpec(BaseModel):
    project_name: str
    endpoints: List[str]
    data_models: List[str]


class AgentState(TypedDict):
    task: str
    spec: TechnicalSpec
    messages: Annotated[List[BaseMessage], "The conversation history"]
    error_log: str
    iteration: int


# --- 3. NODES: The Agent's "Brains" ---

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    anthropic_api_key=os.getenv("ANTROPIC_API_KEY"),
)
llm_with_tools = llm.bind_tools(tools)


def architect_node(state: AgentState):
    structured_llm = llm.with_structured_output(TechnicalSpec)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Lead Architect. Provide a structured Technical Spec for a FastAPI project.",
            ),
            ("user", "{task}"),
        ]
    )
    spec = (prompt | structured_llm).invoke({"task": state["task"]})
    return {"spec": spec, "iteration": 0}


def developer_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an Expert Developer. 
        Your goal: Implement the FastAPI backend in the 'backend/' folder.
        Requirement: Use the 'write_to_disk' tool for EVERY file (main.py, models.py, etc.).
        If there are errors from the tester, fix them and rewrite the files.""",
            ),
            ("user", "Spec: {spec}\nErrors: {error_log}"),
        ]
    )

    # We pass the full spec and any errors
    chain = prompt | llm_with_tools
    response = chain.invoke(
        {"spec": state["spec"], "error_log": state.get("error_log", "None")}
    )

    # We return the message so the ToolNode knows what to do
    return {"messages": [response], "iteration": state["iteration"] + 1}


def tester_node(state: AgentState):
    # For now, a simple logic: if it's the first try, simulate a small error
    # to demonstrate the Fix-Loop. On the second try, it passes.
    if state["iteration"] < 2:
        return {"error_log": "Error: Missing Pydantic validation in main.py"}
    return {"error_log": "PASS"}


# --- 4. GRAPH CONSTRUCTION ---


def should_continue(state: AgentState):
    # If the tester says PASS, we stop. Otherwise, we loop back to Developer.
    if state["error_log"] == "PASS" or state["iteration"] >= 3:
        return "end"
    return "developer"


workflow = StateGraph(AgentState)

workflow.add_node("architect", architect_node)
workflow.add_node("developer", developer_node)
workflow.add_node(
    "write_files", tool_node
)  # Standard ToolNode to execute write_to_disk
workflow.add_node("tester", tester_node)

workflow.set_entry_point("architect")
workflow.add_edge("architect", "developer")
workflow.add_edge("developer", "write_files")
workflow.add_edge("write_files", "tester")

workflow.add_conditional_edges(
    "tester", should_continue, {"developer": "developer", "end": END}
)

app = workflow.compile()

# --- 5. EXECUTION ---

if __name__ == "__main__":
    inputs = {"task": "Create a FastAPI app for a coffee shop with a /menu endpoint"}
    for output in app.stream(inputs):
        print(output)
