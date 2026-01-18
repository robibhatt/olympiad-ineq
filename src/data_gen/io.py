"""I/O utilities for reading and writing JSONL files."""

import json
from pathlib import Path


def read_existing_ids(path: Path | str) -> set[str]:
    """Read existing record IDs from a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        Set of IDs found in the file. Empty set if file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        return set()

    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if "id" in record:
                    ids.add(record["id"])
    return ids


def append_jsonl(path: Path | str, record: dict) -> None:
    """Append a single record to a JSONL file.

    Args:
        path: Path to the JSONL file.
        record: Dictionary to write as a JSON line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
