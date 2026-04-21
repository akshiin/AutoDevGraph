import os

from langchain_core.tools import tool


@tool
def write_to_disk(filename: str, content: str, folder: str = "backend"):
    """Writes code content to a specific file in the local filesystem."""
    full_path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully wrote {filename} to {folder}/"
