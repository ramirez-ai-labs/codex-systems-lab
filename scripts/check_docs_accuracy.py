#!/usr/bin/env python3
"""Fail if any Markdown file references a `python <path>` command or a
relative link pointing at a file that doesn't exist in the repo.

Catches the class of bug where a script gets renamed/moved but the docs
that tell a reader how to run it don't get updated. Stdlib-only so it can
run in CI without installing the project's ML dependencies.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}

RUN_COMMAND_RE = re.compile(r"^\s*python[3]?\s+(\S+\.py)\b", re.MULTILINE)
MD_LINK_RE = re.compile(r"\]\(([^)]+)\)")


def find_markdown_files():
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for filename in filenames:
            if filename.endswith(".md"):
                yield os.path.join(dirpath, filename)


def check_run_commands(md_path, content, errors):
    """A run command is valid if its path resolves either from the repo
    root (e.g. `python 01_inference_profiling/foo.py`) or from the
    Markdown file's own directory (e.g. `python foo.py`, implying a prior
    `cd` into that folder) — both styles are used across this repo."""
    base_dir = os.path.dirname(md_path)
    for match in RUN_COMMAND_RE.finditer(content):
        raw_path = match.group(1)
        from_root = os.path.normpath(os.path.join(REPO_ROOT, raw_path))
        from_file_dir = os.path.normpath(os.path.join(base_dir, raw_path))
        if not os.path.isfile(from_root) and not os.path.isfile(from_file_dir):
            errors.append(
                f"{os.path.relpath(md_path, REPO_ROOT)}: run command references "
                f"missing file '{raw_path}'"
            )


def check_markdown_links(md_path, content, errors):
    base_dir = os.path.dirname(md_path)
    for match in MD_LINK_RE.finditer(content):
        link = match.group(1).strip()
        if not link or link.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path_part = link.split("#", 1)[0]
        if not path_part:
            continue
        candidate = os.path.normpath(os.path.join(base_dir, path_part))
        if not os.path.exists(candidate):
            errors.append(
                f"{os.path.relpath(md_path, REPO_ROOT)}: link references missing "
                f"path '{link}'"
            )


def main():
    errors = []
    md_files = list(find_markdown_files())

    for md_path in md_files:
        with open(md_path, encoding="utf-8") as f:
            content = f.read()
        check_run_commands(md_path, content, errors)
        check_markdown_links(md_path, content, errors)

    if errors:
        print(f"Found {len(errors)} docs accuracy issue(s):\n")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Checked {len(md_files)} markdown files — all run commands and links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
