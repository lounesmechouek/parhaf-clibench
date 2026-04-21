# Docker and image tags

The package ships two Dockerfiles under `infra/docker/` and a
GitHub Actions workflow that builds and pushes one of them to
DockerHub on every merge to `main`.

## The images

| Dockerfile                    | Purpose                                                                   |
|-------------------------------|---------------------------------------------------------------------------|
| `infra/docker/Dockerfile.vllm`| vLLM 0.8.5 runtime plus the benchmark code, ready to serve a model.      |
| `infra/docker/Dockerfile.bench`| Benchmark-only image without vLLM, for scoring and UI in a CI context.   |

The vLLM image is the one the CD pipeline builds. It carries SSH and
rsync so RunPod pods can be driven remotely.

## Tagging scheme

Each push to `main` publishes two tags:

| Tag                                   | Stability                                                |
|---------------------------------------|----------------------------------------------------------|
| `sha-<short>`                          | Immutable. Always rebuilds from the same commit.         |
| `main`                                | Rolling. Updated on every merge to `main`.              |

The immutable tag is the reproducibility anchor. Production pipelines
pin this tag. The rolling tag is for developer convenience.

Both tags are published to the repository specified by
`vars.DOCKER_IMAGE_REPO` on the CI runner, or to
`${DOCKERHUB_USERNAME}/parhaf-clinbench-vllm` when the repo variable
is unset.

## Building locally

```bash
docker buildx build \
  --file infra/docker/Dockerfile.vllm \
  --platform linux/amd64 \
  --tag parhaf-clinbench-vllm:local \
  .
```

The build assumes a BuildKit-enabled Docker daemon. A full build
from cold cache takes roughly 15 minutes on a modern laptop and
produces a ~12 GB image because of the vLLM wheels.

## Pulling a published image

```bash
docker pull <registry>/parhaf-clinbench-vllm:sha-<short>
```

Replace the registry and the tag with the values from your release.
The `main` rolling tag is only appropriate for development.

## Entrypoints

Three helper shell scripts live under `infra/docker/entrypoints/`:

| Script                          | Behavior                                                         |
|---------------------------------|------------------------------------------------------------------|
| `run_vllm_then_benchmark.sh`    | Start vLLM, wait for healthcheck, run the benchmark, export.    |
| `run_benchmark.sh`              | Run the benchmark against an already-running vLLM.              |
| `export_and_exit.sh`            | Collect results, push them to the configured storage, exit.     |

On RunPod, the first one is the default: a pod becomes a
self-contained benchmark worker that stops automatically when the
suite finishes.

## The CD pipeline

The `.github/workflows/docker.yml` workflow:

1. Frees disk space on the GitHub-hosted runner.
2. Resolves the image reference from secrets and repository
   variables.
3. Logs in to DockerHub.
4. Builds with `docker/build-push-action@v6` and pushes both tags.
5. Uses a registry-backed build cache to keep rebuilds under a few
   minutes.

Required secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Optional repository variable:

- `DOCKER_IMAGE_REPO`, to override the default `<user>/parhaf-clinbench-vllm`.

The workflow fails fast if a secret is missing, so a misconfiguration
is visible on the pull request that introduces it.
