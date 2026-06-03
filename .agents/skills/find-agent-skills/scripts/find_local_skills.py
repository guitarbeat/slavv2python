from __future__ import annotations

import argparse
from pathlib import Path


def read_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    _, block, _ = text.split("---", 2)
    result: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"')
    return result


def score(query: str, name: str, description: str, body: str) -> int:
    terms = [term.lower() for term in query.split() if term.strip()]
    haystack = f"{name} {description} {body}".lower()
    return sum(1 for term in terms if term in haystack)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search project-local agent skills.")
    parser.add_argument("query", nargs="*", help="Search terms.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(".agents") / "skills",
        help="Skills directory to scan.",
    )
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    rows: list[tuple[int, str, str, Path]] = []
    for skill_file in sorted(args.root.glob("*/SKILL.md")):
        body = skill_file.read_text(encoding="utf-8")
        metadata = read_frontmatter(skill_file)
        name = metadata.get("name", skill_file.parent.name)
        description = metadata.get("description", "")
        rank = score(query, name, description, body) if query else 1
        if rank or not query:
            rows.append((rank, name, description, skill_file))

    rows.sort(key=lambda row: (-row[0], row[1]))
    for _, name, description, path in rows:
        print(f"{name}\t{path}\t{description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
