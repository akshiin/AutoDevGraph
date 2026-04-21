from typing import List, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


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
