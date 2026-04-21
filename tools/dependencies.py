import subprocess

from langchain_core.tools import tool


@tool
def add_uv_dependency(package: str) -> str:
    """Adds a Python package dependency to the project using uv."""
    result = subprocess.run(
        ["uv", "add", package],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"Failed to add '{package}': {result.stderr.strip()}"
    return f"Successfully added '{package}' via uv.\n{result.stdout.strip()}"


@tool
def remove_uv_dependency(package: str) -> str:
    """Removes a Python package dependency from the project using uv."""
    result = subprocess.run(
        ["uv", "remove", package],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return f"Failed to remove '{package}': {result.stderr.strip()}"
    return f"Successfully removed '{package}' via uv.\n{result.stdout.strip()}"
