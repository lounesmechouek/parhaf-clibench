"""Streamlit entry point for the benchmark results explorer.

This shell assembles the presentation pages used to review a full
PARHAF-LM-CLINBENCH run. The app is intentionally thin: all clinical
benchmark logic stays in ``src/parhaf_clinbench`` while this module only
handles navigation and Streamlit bootstrap concerns.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import streamlit as st

PageRenderer = Callable[[], None]


def _ensure_repo_root_on_path() -> None:
    """Make local ``ui`` imports work when Streamlit launches this file directly."""

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _pages() -> dict[str, PageRenderer]:
    """Load page renderers lazily after bootstrapping the repository root."""

    _ensure_repo_root_on_path()

    from ui.theme import inject_css
    from ui.views.error_explorer import render as render_errors
    from ui.views.head_to_head import render as render_h2h
    from ui.views.leaderboard import render as render_leaderboard
    from ui.views.methodology import render as render_methodology
    from ui.views.model_card import render as render_model
    from ui.views.overview import render as render_overview
    from ui.views.robustness import render as render_robustness
    from ui.views.subgroups import render as render_subgroups
    from ui.views.task_deep_dive import render as render_task

    inject_css()
    return {
        "🏠 Overview": render_overview,
        "🏆 Leaderboard": render_leaderboard,
        "🔬 Task deep dive": render_task,
        "🧬 Model card": render_model,
        "⚔️ Head-to-head": render_h2h,
        "🛡️ Robustness": render_robustness,
        "🧪 Subgroups": render_subgroups,
        "❗ Error explorer": render_errors,
        "📖 Methodology": render_methodology,
    }


def main() -> None:
    """Render the navigation shell and dispatch to the selected page."""

    st.set_page_config(
        page_title="SLM Benchmark on Clinical Notes",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = _pages()
    st.sidebar.title("PARHAF-CLINBENCH")
    choice = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Bootstrap: B = 1000 · percentile 95% CI · document-level.\n\n"
        "Lounes Mechouek - AI/ML Engineer\nmechouek.contact@gmail.com"
    )
    st.sidebar.markdown(
        """
        <div style="display:flex; gap:12px; align-items:center;">
            <a href="https://github.com/lounesmechouek/parhaf-clibench"
               target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                <img src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/github.svg"
                     width="24" alt="GitHub logo">
            </a>
            <a href="https://www.linkedin.com/in/lounes-mechouek/"
               target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                <img src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/linkedin.svg"
                     width="24" alt="LinkedIn logo">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pages[choice]()


if __name__ == "__main__":
    main()
