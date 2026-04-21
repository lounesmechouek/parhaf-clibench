# Datasets and tasks

A **task** is the unit of evaluation. Each task describes a
transformation from a free-text clinical note into a list of
structured records. Four tasks ship with the package, matching the
four annotated subsets of PARHAF.

## The four tasks

| Task         | Target                                                  | Official metric                        | HuggingFace subset                                |
|--------------|---------------------------------------------------------|----------------------------------------|---------------------------------------------------|
| `pseudo`     | Patient identifiers with exact character offsets         | `span_micro_f1`                        | `HealthDataHub/PARHAF-pseudo-annotated`           |
| `infectio`   | Bacteria, infections, anatomical sites with negation     | `text_label_negation_micro_f1`         | `HealthDataHub/PARHAF-infectiology-annotated`     |
| `response`   | Treatment-outcome classification with justification span | `text_label_micro_f1`                  | `HealthDataHub/PARHAF-response_to_treatment-annotated` |
| `scenario`   | Eight structured patient fields                          | `text_label_micro_f1`                  | `HealthDataHub/PARHAF`                            |

The metrics registry lives in
[`parhaf_clinbench.tasks`](../reference/parhaf_clinbench/tasks/index.md)
and is the single source of truth for what counts as "the" score of a
task.

## Task configuration

Each task has a YAML file under `configs/tasks/`:

```yaml title="configs/tasks/pseudo.yaml"
task_id: pseudo
dataset: HealthDataHub/PARHAF-pseudo-annotated
dataset_revision: 4d866f075a4d91c4e5ce0058feedd6da1d8e879a
text_field: full_text
label_field: span_type
official_metric: span_micro_f1
required_record_fields: [label, text, start, end]
```

The `dataset_revision` field pins the HuggingFace commit SHA that the
loader will download. This pin is what makes the task reproducible
over time.

## The canonical record schema

Every task resolves to the same Pydantic model:
`parhaf_clinbench.core.models.CanonicalDocument`, holding a list of
`Record` objects. The fields required by each task are declared in
the task YAML (`required_record_fields`) and validated on every run.

```python
class Record(BaseModel):
    label: str
    text: str
    start: int | None = None
    end: int | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
```

`attributes` carries task-specific metadata. For `infectio`, it
includes `negation` with values in `{Present, Absent, Indetermine}`.
For `pseudo`, it can include `role` distinguishing patient identifiers
from staff identifiers.

## Dataset loading and caching

The loader in
[`parhaf_clinbench.data.hf_loaders`](../reference/parhaf_clinbench/data/hf_loaders.md)
tries a local cache first. If the cache contains the pinned revision
and fingerprint, it runs without any network I/O. The fingerprint is
a SHA-256 over the dataset name, example count, document IDs, and
text hashes, computed by
[`parhaf_clinbench.data.manifests`](../reference/parhaf_clinbench/data/manifests.md).

!!! tip "Offline runs"
    Run `parhaf-clinbench prefetch-suite --suite <suite>` once on
    a machine with network access. After that, the benchmark can run
    indefinitely offline against the same pinned revisions.

## Known limitations

- Tasks are not pluggable through entry points today. Adding a new
  task means adding a new schema, a new scorer, and a new loader
  branch. See the [Contributing page](../about/contributing.md) for
  the scope of that change.
- The scoring metrics are locked to the official ones above. If you
  need a relaxed metric (for example text-only matching on `pseudo`),
  compute it from the predictions JSONL rather than from inside the
  scorer.
