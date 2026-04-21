# LLMOps

LLMOps is the set of engineering practices that keep a language-model
benchmark trustworthy over time. For `parhaf-clinbench` this covers
six questions that matter more for LLMs than for classical ML:

- What exactly ran? Versioning and hashing of models, datasets,
  prompts, and code.
- How do we prove it ran? Run manifests that capture every pin and
  parameter.
- Is it deterministic enough? Which knobs are greedy, which are not.
- Can we re-verify the numbers? The independent scoring audit.
- Is the code allowed to land? The CI pipeline and its gates.
- How do we run it in production? The Docker image, the RunPod flow,
  and the artifact layout.

These are covered one at a time in the pages below.

| Page                                                | Covers                                                    |
|-----------------------------------------------------|-----------------------------------------------------------|
| [Versioning and hashing](versioning.md)             | What the package versions, and how.                       |
| [Run manifests](manifests.md)                       | The `manifest.json` schema and a worked example.          |
| [Determinism](determinism.md)                       | What is reproducible and what is not, with exact knobs.   |
| [Scoring audit](scoring-audit.md)                   | The independent rescoring pipeline.                       |
| [CI pipeline](ci.md)                                | What the pull-request gate guarantees.                    |
| [Docker and image tags](docker.md)                  | The images published to DockerHub.                        |
| [RunPod and GPU pods](runpod.md)                    | The cloud deployment path.                                |
| [Artifacts and storage](artifacts.md)               | The on-disk contract for a run.                           |

## The philosophy

A reproducibility claim is only as strong as the weakest pin. A run
that reports a fixed seed but not a dataset revision is not
reproducible. A run that pins the dataset but not the prompt is not
reproducible either. The goal of every page in this section is to
leave no object unpinned.
