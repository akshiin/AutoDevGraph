from state import AgentState


def tester_node(state: AgentState) -> dict:
    # Simulates a fix-loop: returns an error on the first pass, PASS on subsequent ones.
    if state["iteration"] < 2:
        return {"error_log": "Error: Missing Pydantic validation in main.py"}
    return {"error_log": "PASS"}
