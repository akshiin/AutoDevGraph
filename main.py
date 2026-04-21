import os

from typing import List, TypedDict, Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

# --- 1. TOOLS: The Agent's "Hands" ---


@tool
def write_to_disk(filename: str, content: str, folder: str = "backend"):
    """Writes code content to a specific file in the local filesystem."""
    full_path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote {filename} to {folder}/"


tools = [write_to_disk]
tool_node = ToolNode(tools)

# --- 2. STATE & SCHEMAS ---


class TechnicalSpec(BaseModel):
    project_name: str
    endpoints: List[str]
    data_models: List[str]
    crud_operations: List[str]
    validation_rules: List[str]
    business_logic: List[str]
    required_files: List[str]


class AgentState(TypedDict):
    task: str
    spec: TechnicalSpec
    messages: Annotated[List[BaseMessage], add_messages]
    error_log: str
    iteration: int


# --- 3. NODES: The Agent's "Brains" ---

# llm = ChatAnthropic(
#     model="claude-3-5-sonnet-20240620",
#     temperature=0,
#     anthropic_api_key=os.getenv("ANTROPIC_API_KEY"),
# )
llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)


def architect_node(state: AgentState):
    structured_llm = llm.with_structured_output(TechnicalSpec)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Senior System Architect specializing in FastAPI and Clean Architecture.
                Your job is to analyze a project description and produce a complete, concrete Technical Specification \
                that a Developer Agent will turn directly into runnable code.

                Your spec MUST cover every section below. Be exhaustive and specific to the project described — \
                derive all models, endpoints, and rules from the requirements, do not invent generic examples.

                DATA ACCESS LAYER (SQLAlchemy):
                - Identify every ORM model the project needs. For each model list all column names, Python types, \
                constraints (nullable, unique, default), and relationships (ForeignKey, one-to-many, etc.).
                - Define every CRUD function required (name, parameters, return type, and a one-line description \
                of what it does).

                VALIDATION RULES (Pydantic):
                - For every field that has a format constraint, specify the exact regex or validator to use.
                - For any date/age constraint, describe the arithmetic check precisely.
                - For email fields, use EmailStr or a documented regex mask.

                BUSINESS LOGIC:
                - Document every domain rule (limits, ordering, state machines, computed fields, etc.) with \
                enough detail that a developer can implement it without ambiguity.
                - For any security requirement (hashing, token generation, protected routes), specify the \
                exact library and algorithm to use.

                REQUIRED FILES — always list these as the target file structure:
                backend/database.py, backend/models.py, backend/schemas.py, backend/auth_utils.py,
                backend/routers/<router_name>.py (one file per logical domain derived from the project),
                backend/main.py

                ENDPOINTS — for every endpoint specify: HTTP method, path, which router file it belongs to, \
                request schema, response schema, auth requirement, and any business-rule enforced.""",
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
                """You are a Senior Backend Developer. Your output is production-grade Python code.

ABSOLUTE RULES — violating any of these is a failure:
1. NEVER write placeholders, `pass` statements, `# TODO`, `# implementation here`, or any stub.
2. Every function must contain complete, working logic — no exceptions.
3. Use SQLAlchemy ORM (not raw SQL) for ALL database operations. Database: SQLite file `backend/database.db`.
4. If the spec requires password hashing, use passlib CryptContext(schemes=["bcrypt"]).
5. If the spec requires JWT authentication, use python-jose (HS256, SECRET_KEY from env, 30-min expiry).
6. Protect routes that require auth with OAuth2PasswordBearer + a `get_current_user` dependency.
7. Implement every validation rule and business constraint exactly as described in the spec.

FILE STRUCTURE — write each file in order using write_to_disk, one call per file:
- database.py    → SQLAlchemy engine (sqlite:///./backend/database.db), SessionLocal, Base, get_db dependency
- models.py      → All SQLAlchemy ORM models derived from the spec
- schemas.py     → All Pydantic request/response schemas with full validation as described in the spec
- auth_utils.py  → All auth helpers required by the spec (hashing, token creation, current-user dependency)
- routers/<name>.py → One file per logical domain from the spec; complete endpoint logic, no stubs
- main.py        → FastAPI app factory: include all routers, call Base.metadata.create_all on startup

PROCESS:
1. Write database.py first, then models.py, schemas.py, auth_utils.py.
2. Write each router file with complete endpoint logic.
3. Write main.py last.
4. When all files are written, stop calling tools and respond with a plain summary message.
5. If errors are provided, identify the root cause and rewrite only the affected files.""",
            ),
            ("user", "Spec: {spec}\n\nErrors from tester: {error_log}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm_with_tools
    response = chain.invoke(
        {
            "spec": state["spec"],
            "error_log": state.get("error_log", "None"),
            "messages": state.get("messages", []),
        }
    )

    return {"messages": [response], "iteration": state["iteration"] + 1}


def tester_node(state: AgentState):
    # For now, a simple logic: if it's the first try, simulate a small error
    # to demonstrate the Fix-Loop. On the second try, it passes.
    if state["iteration"] < 2:
        return {"error_log": "Error: Missing Pydantic validation in main.py"}
    return {"error_log": "PASS"}


# --- 4. GRAPH CONSTRUCTION ---


def should_use_tools(state: AgentState):
    """Route developer output: if the agent made tool calls, execute them;
    otherwise it is done writing and we hand off to the tester."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "write_files"
    return "tester"


def should_continue(state: AgentState):
    if state["error_log"] == "PASS" or state["iteration"] >= 3:
        return "end"
    return "developer"


workflow = StateGraph(AgentState)

workflow.add_node("architect", architect_node)
workflow.add_node("developer", developer_node)
workflow.add_node("write_files", tool_node)
workflow.add_node("tester", tester_node)

workflow.set_entry_point("architect")
workflow.add_edge("architect", "developer")

# ReAct loop: developer ↔ write_files until the agent stops making tool calls
workflow.add_conditional_edges(
    "developer", should_use_tools, {"write_files": "write_files", "tester": "tester"}
)
workflow.add_edge("write_files", "developer")

workflow.add_conditional_edges(
    "tester", should_continue, {"developer": "developer", "end": END}
)

app = workflow.compile()

# --- 5. EXECUTION ---

if __name__ == "__main__":

    user_task = """
    **Role**: You are a Senior System Architect specializing in FastAPI and Clean Architecture.
    **Goal**: Transform the provided "Portfolio Platform" project description into a structured technical specification for a Developer Agent.

    **Instructions**:
    Analyze the project description and output a JSON or Markdown technical specification containing the following sections:

    1. **Database Schema**:
    - Define a 'User' (Author) model with all attributes (First Name, Last Name, Username, Profession, Birthdate, Email, Password Hash).
    - Define a 'SocialLink' model (One-to-Many with User).
    - Define a 'Project' model with attributes for Description, Cover S3 URL, and Visibility toggles.
    - Define a 'Slide' model (One-to-Many with Project) with an 'order_index' and S3 image paths.

    2. **Validation Rules (Pydantic)**:
    - Create strict regex patterns for Name/Username (no spaces, no brackets, no punctuation).
    - Implement the "12+ Age" logic (Birthdate validation).
    - Implement Email mask validation.

    3. **API Endpoints (FastAPI)**:
    - `POST /auth/register`: Handle registration, password generation, and mock email trigger.
    - `POST /auth/login`: JWT-based authentication.
    - `GET/PUT /user/profile`: Personal data and Social Links management.
    - `GET/POST/PUT /project`: Handle the single-project limitation (Max 1 project per user).
    - `POST /project/slides`: Logic for max 12 slides and ordering.
    - `GET /view/{protected_link}`: Public-facing view with visibility logic.

    4. **Business Logic & Constraints**:
    - Random 8-character password generation logic.
    - S3 Integration: Define logic for private S3 storage for images.
    - Project logic: Enforce "Step-by-step" tab progression (Description -> Content -> Publication).

    5. **Security**:
    - Password hashing (passlib/bcrypt).
    - Protected routes using OAuth2 (JWT).

    **Output Format**: 
    Provide a clear "Contract" that a developer can use to create models.py, schemas.py, and main.py. Do not write the code yourself; write the detailed architectural instructions.
    """
    inputs = {"task": user_task}
    for output in app.stream(inputs):
        print(output)
