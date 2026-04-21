# Changelog

The project follows [semantic versioning](https://semver.org). Each
release is tagged on GitHub and published as a Docker image on
DockerHub. Immutable image tags map one-to-one to commit SHAs.

## 0.1.0

Initial public release.

- Four PARHAF tasks: pseudonymization, infectiology, response to
  treatment, structured scenario.
- Two tracks: zero-shot and few-shot fixed.
- vLLM runtime with xgrammar-backed structured output.
- GLiNER2 encoder runtime with heuristic negation.
- Mock runtime for CI.
- Document-level non-parametric bootstrap with 1,000 replications.
- Paired bootstrap for head-to-head comparisons.
- Robustness metrics: schema conformity, empty rate, JSON validity,
  latency.
- Error taxonomy with document-level drill-down.
- Streamlit UI with nine pages.
- Rich-based terminal monitoring dashboard.
- Docker image published on DockerHub.
- RunPod ops helpers.
- Full scoring audit with six-decimal tolerance.

Supporting infrastructure:

- `uv` as the package manager.
- Google-style docstrings throughout `src/`.
- `mypy --strict` across the source tree.
- 70% coverage floor on the pull-request gate.
