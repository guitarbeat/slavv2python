from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(".")
SKILLS_ROOT = ROOT / ".agents" / "skills"
TODO_PATH = ROOT / "docs" / "TODO.md"
FINDINGS_PATH = ROOT / "docs" / "reference" / "core" / "EXACT_PROOF_FINDINGS.md"
SOLUTIONS_ROOT = ROOT / "docs" / "solutions"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    evidence: str
    reason: str
    target: str


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def frontmatter_value(text: str, key: str) -> str:
    if not text.startswith("---"):
        return ""
    block = text.split("---", 2)[1]
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", block, flags=re.MULTILINE)
    return match.group(1).strip().strip('"') if match else ""


def load_skills() -> list[Skill]:
    skills: list[Skill] = []
    for skill_file in sorted(SKILLS_ROOT.glob("*/SKILL.md")):
        text = read_text(skill_file)
        skills.append(
            Skill(
                name=frontmatter_value(text, "name") or skill_file.parent.name,
                description=frontmatter_value(text, "description"),
                body=text,
                path=skill_file,
            )
        )
    return skills


def has_skill(skills: list[Skill], *terms: str) -> bool:
    haystack = "\n".join(f"{skill.name} {skill.description} {skill.body}" for skill in skills).lower()
    return all(term.lower() in haystack for term in terms)


def skill_body_contains(skills: list[Skill], skill_name: str, phrase: str) -> bool:
    for skill in skills:
        if skill.name == skill_name:
            return phrase.lower() in skill.body.lower()
    return False


def active_checkboxes(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("- [ ]")]


def active_blockers(text: str) -> list[str]:
    in_blockers = False
    blockers: list[str] = []
    for line in text.splitlines():
        if line.startswith("## ") and "Active blockers" in line:
            in_blockers = True
            continue
        if in_blockers and line.startswith("## "):
            break
        if in_blockers and re.match(r"\d+\.\s+\*\*", line):
            blockers.append(line.strip())
    return blockers


def solution_count() -> int:
    if not SOLUTIONS_ROOT.exists():
        return 0
    return sum(1 for path in SOLUTIONS_ROOT.rglob("*.md") if path.name.lower() != "readme.md")


def build_candidates(skills: list[Skill]) -> list[Candidate]:
    todo = read_text(TODO_PATH)
    findings = read_text(FINDINGS_PATH)
    tasks = active_checkboxes(todo)
    blockers = active_blockers(findings)
    candidates: list[Candidate] = []

    parity_evidence = "; ".join((tasks + blockers)[:3])
    if parity_evidence and not has_skill(skills, "parity", "proof"):
        candidates.append(
            Candidate(
                name="parity-proof-compounder",
                kind="new skill",
                evidence=parity_evidence,
                reason=(
                    "Active parity work repeatedly asks agents to check run status, "
                    "prove the first failing stage, record findings, and promote verified fixes."
                ),
                target=".agents/skills/parity-proof-compounder/SKILL.md",
            )
        )
    elif parity_evidence and not skill_body_contains(
        skills,
        "self-improving-agent",
        "For parity proof sessions",
    ):
        candidates.append(
            Candidate(
                name="strengthen existing parity skills",
                kind="skill edit",
                evidence=parity_evidence,
                reason=(
                    "Parity is already covered by docs and prompts, but the generative promotion step "
                    "should explicitly create solution notes or skill edits after verified failures."
                ),
                target=".agents/skills/self-improving-agent/SKILL.md",
            )
        )

    if solution_count() <= 1 and not has_skill(skills, "solution", "note"):
        candidates.append(
            Candidate(
                name="solution-note-generator",
                kind="new skill",
                evidence=f"Only {solution_count()} documented solution note(s) found under docs/solutions.",
                reason=(
                    "The docs describe a compound-solutions workflow, but there is little searchable "
                    "solution history relative to the amount of parity and integration debugging."
                ),
                target=".agents/skills/solution-note-generator/SKILL.md",
            )
        )
    elif solution_count() <= 1:
        candidates.append(
            Candidate(
                name="exercise solution-note-generator",
                kind="scratch reflection",
                evidence=f"Only {solution_count()} documented solution note(s) found, but a solution-note skill exists.",
                reason=(
                    "The next generative step is to use the skill after the next verified parity or "
                    "integration fix, then check whether its template captured enough evidence."
                ),
                target="workspace/scratch/agent-reflections.md",
            )
        )

    if (
        any("monitor" in task.lower() or "rerun" in task.lower() for task in tasks)
        and not skill_body_contains(skills, "control-cli", "SLAVV Run Ops")
    ):
        candidates.append(
            Candidate(
                name="run-ops-watch",
                kind="skill edit",
                evidence="Active tasks mention reruns, PIDs, duplicate writers, and monitor/status commands.",
                reason=(
                    "Run operations are a recurring workflow; strengthen control-cli or add a narrow "
                    "run-monitoring workflow only if agents keep missing the PID/duplicate-writer guard."
                ),
                target=".agents/skills/control-cli/SKILL.md",
            )
        )

    if (
        has_skill(skills, "workflow", "chats")
        and has_skill(skills, "self", "improving")
        and not skill_body_contains(skills, "find-agent-skills", "Generative Skill Loop")
    ):
        candidates.append(
            Candidate(
                name="preflight skill selection",
                kind="skill edit",
                evidence="Both workflow-from-chats and self-improving-agent exist.",
                reason=(
                    "Before creating any new skill, agents should run local skill discovery and decide "
                    "whether the evidence belongs in an existing skill, a solution note, or scratch."
                ),
                target=".agents/skills/find-agent-skills/SKILL.md",
            )
        )

    return candidates


def render_markdown(candidates: list[Candidate]) -> str:
    if not candidates:
        return "No generative skill candidates found.\n"
    lines = ["# Generative Skill Candidates", ""]
    for candidate in candidates:
        lines.extend(
            [
                f"## {candidate.name}",
                "",
                f"- Kind: {candidate.kind}",
                f"- Evidence: {candidate.evidence}",
                f"- Reason: {candidate.reason}",
                f"- Target: `{candidate.target}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Suggest generative skill candidates from repo signals.")
    parser.add_argument("--format", choices=["markdown"], default="markdown")
    args = parser.parse_args()

    candidates = build_candidates(load_skills())
    if args.format == "markdown":
        print(render_markdown(candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
