def truncate_delta(diff_output: str, max_lines: int = 200) -> str:
    """Truncate diff to max_lines with indicator if truncated."""
    lines = diff_output.splitlines()
    if len(lines) <= max_lines:
        return diff_output

    truncated = "\n".join(lines[:max_lines])
    remaining = len(lines) - max_lines
    return f"{truncated}\n\n... ({remaining} lines omitted)"
