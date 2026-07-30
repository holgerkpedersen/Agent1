from typing import List, Dict, Any


def generate_prompt_with_tools(issue_description: str) -> List[Dict[str, Any]]:
    """Generate prompt that explicitly requests native tool usage."""
    system_content = (
        "You are FixCommand - an intelligent code repair assistant. "
        "Use the provided tools (`read_file`, `apply_fix`) directly when identifying issues or applying fixes. "
        "Do not use text markers, regex patterns, or custom delimiters to represent tool calls. "
        "Always invoke tools natively through the SDK's function-calling mechanism."
    )

    user_content = (
        f"Issue Description: {issue_description}\n\n"
        "Available functions:\n"
        "- read_file: Read the contents of a file by filename.\n"
        "- apply_fix: Apply a patch to a specific line in a file."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
