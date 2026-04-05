"""Prompt rendering utilities for Jinja2 templates."""

from __future__ import annotations

from pathlib import Path

from parhaf_clinbench.core.enums import TaskId, TrackId
from parhaf_clinbench.prompting.contracts import prompt_dynamic_context
from parhaf_clinbench.prompting.registry import prompt_template_path


def render_prompt(
    *,
    task: TaskId,
    track: TrackId,
    document_id: str,
    text: str,
    fewshot_examples: str,
    speciality_metadata: str | None = None,
) -> str:
    """Render a prompt from the template associated with task and track.

    Args:
        task: Task identifier.
        track: Evaluation track.
        document_id: Source document identifier.
        text: Source document text.
        fewshot_examples: Pre-rendered few-shot examples block.
        speciality_metadata: Optional speciality-specific context.

    Returns:
        Fully rendered prompt string.

    Examples:
        >>> isinstance(
        ...     render_prompt(
        ...         task=TaskId.PSEUDO,
        ...         track=TrackId.ZEROSHOT,
        ...         document_id="doc-1",
        ...         text="example text",
        ...         fewshot_examples="",
        ...     ),
        ...     str,
        ... )
        True
    """

    template_file = prompt_template_path(task=task, track=track)
    content = Path(template_file).read_text(encoding="utf-8")
    dynamic_context = prompt_dynamic_context(
        task=task,
        speciality_metadata=speciality_metadata,
    )
    try:
        from jinja2 import Template

        template = Template(content)
        return template.render(
            document_id=document_id,
            text=text,
            fewshot_examples=fewshot_examples,
            **dynamic_context,
        )
    except Exception:
        rendered = content.replace("{{ document_id }}", document_id)
        rendered = rendered.replace("{{ text }}", text)
        rendered = rendered.replace("{{ fewshot_examples }}", fewshot_examples)
        rendered = rendered.replace(
            "{{ canonical_schema_json }}",
            str(dynamic_context.get("canonical_schema_json", "")),
        )
        rendered = rendered.replace(
            "{{ offset_policy }}",
            str(dynamic_context.get("offset_policy", "")),
        )
        rendered = rendered.replace(
            "{{ speciality_metadata }}",
            str(dynamic_context.get("speciality_metadata", "")),
        )
        return rendered
