from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import llm
from state import AgentState
from tools import all_tools

_SYSTEM_PROMPT = """You are a Senior Backend Developer. Your output is production-grade Python code.

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
5. If errors are provided, identify the root cause and rewrite only the affected files."""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("user", "Spec: {spec}\n\nErrors from tester: {error_log}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

_llm_with_tools = llm.bind_tools(all_tools)


def developer_node(state: AgentState) -> dict:
    chain = _prompt | _llm_with_tools
    response = chain.invoke(
        {
            "spec": state["spec"],
            "error_log": state.get("error_log", "None"),
            "messages": state.get("messages", []),
        }
    )
    return {"messages": [response], "iteration": state["iteration"] + 1}
