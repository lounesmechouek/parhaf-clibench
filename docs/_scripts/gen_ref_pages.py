"""Generate API reference pages for the `parhaf_clinbench` package.

This script is invoked by the `mkdocs-gen-files` plugin during the docs
build. It walks the `src/parhaf_clinbench` tree and emits one Markdown
page per public submodule under `docs/reference/`, each containing a
single `::: parhaf_clinbench.<module>` directive that `mkdocstrings`
expands into a documented API page.

Private modules (any segment starting with an underscore) are skipped.

Typical usage is not to run this script directly, but to invoke `mkdocs`
which loads it through the `gen-files` plugin:

    uv run --extra docs mkdocs serve
    uv run --extra docs mkdocs build --strict
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

SRC_ROOT = Path("src")
PACKAGE = "parhaf_clinbench"
REFERENCE_ROOT = Path("reference")

nav = mkdocs_gen_files.Nav()

for path in sorted((SRC_ROOT / PACKAGE).rglob("*.py")):
    module_path = path.relative_to(SRC_ROOT).with_suffix("")
    parts = tuple(module_path.parts)

    if any(part.startswith("_") and part != "__init__" for part in parts):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue
        doc_path = REFERENCE_ROOT / Path(*parts) / "index.md"
        nav_parts = parts[1:] if len(parts) > 1 else ()
    else:
        doc_path = REFERENCE_ROOT / Path(*parts).with_suffix(".md")
        nav_parts = parts[1:]

    if nav_parts:
        nav[nav_parts] = doc_path.relative_to(REFERENCE_ROOT).as_posix()
    else:
        nav["Overview"] = doc_path.relative_to(REFERENCE_ROOT).as_posix()

    identifier = ".".join(parts)
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"# `{identifier}`\n\n")
        fd.write(f"::: {identifier}\n")

with mkdocs_gen_files.open(REFERENCE_ROOT / "SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
