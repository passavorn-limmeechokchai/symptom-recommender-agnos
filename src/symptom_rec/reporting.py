"""Update one generated section of REPORT.md without replacing other report sections."""
import fcntl
import os
import re
import tempfile
from pathlib import Path

from .paths import REPO_ROOT

REPORT_FILE = REPO_ROOT / "REPORT.md"
TITLES = {
    "evaluation": "Default recommender and normalizer evaluation",
    "retrieval-development": "E5-base retrieval development and limitations",
    "retrieval-evaluation": "E5-base retrieval evaluation",
    "semantic-validation": "E5-small training calibration",
    "semantic-evaluation": "E5-small evaluation",
    "hitrate-diagnostics": "HitRate diagnostics and statistical significance",
    "dedup-diagnostics": "Duplicates, shuffling and normalization metrics",
    "normalization-assertions": "Assertion handling and review benchmark (0.2.0 checkpoint)",
    "closeout": "Runtime verification and clinical review readiness (0.3.0)",
    "prospective-evaluation": "Prospective evaluation protocol and results",
}


def section_markers(name):
    if name not in TITLES:
        raise ValueError("Unknown report section")
    return f"<!-- report:{name}:start -->", f"<!-- report:{name}:end -->"


def render_section(name, content):
    start, end = section_markers(name)
    lines = content.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    # The section owns its level-two heading; demote headings in script-generated content.
    body = "\n".join(re.sub(r"^(#{1,5}) ", r"\1# ", line) for line in lines).strip()
    return f'{start}\n<a id="{name}"></a>\n\n## {TITLES[name]}\n\n{body}\n{end}'


def write_report_section(name, content, path=REPORT_FILE):
    """Lock, read, replace one block and atomically publish; malformed markers fail closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start, end = section_markers(name)
    block = render_section(name, content)
    lock_path = path.with_name("." + path.name + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        original = path.read_text(encoding="utf-8") if path.exists() else "# Evaluation report\n"
        if original.count(start) != original.count(end) or original.count(start) > 1:
            raise ValueError("Malformed report section markers; report was not changed")
        if start in original:
            begin, finish = original.index(start), original.index(end)
            if finish < begin:
                raise ValueError("Reversed report section markers; report was not changed")
            updated = original[:begin] + block + original[finish + len(end):]
        else:
            updated = original.rstrip() + "\n\n" + block + "\n"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix="." + path.name + ".", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(updated)
            mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
            temporary.chmod(mode)
            os.replace(temporary, path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
    return path
