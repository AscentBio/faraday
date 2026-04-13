# Faraday Agent

### Accelerating science with agentic AI for research and drug design

[Faraday](https://ascentbio.ai/) is an AI scientist that carries out research workflows end to end: it reasons over your goals, pulls in uploaded files and external context, and uses specialized tools spanning structural biology, chemistry, ADME, toxicity, and related domains. [Ascent Bio](https://ascentbio.ai/) builds Faraday as a **unified platform** so teams can analyze data, explore SAR and competitive landscapes, and move faster from hypothesis to insight—with reproducible outputs you can review and extend.

This repository is the **open-source agent runtime** (`faraday-oss`): the same agent-oriented stack behind the product, packaged for local or self-hosted use.

---

**Researchers** get a capable collaborator that works autonomously on a task until it is complete, using data and tools along the way, so you can focus on reviewing findings and follow-ups.

**Developers** get a Python package and CLI to run the agent with configurable models, memory backends, and YAML-based runtime settings—suitable for integration into your own infra or experiments.

**Unified context** means conversation, file corpora, retrieval (RAG), and tool outputs are wired together so the model sees one coherent workspace rather than disconnected snippets.

---

For the hosted product (start in the browser, team features, and enterprise-ready deployment), use **[platform.ascentbio.ai](https://platform.ascentbio.ai/)**. Request access via **[accounts.platform.ascentbio.ai](https://accounts.platform.ascentbio.ai)** when needed.

---

## Install

From the repo root (development install):

```bash
pip install -e .
```

Or install from a built wheel/source distribution if you publish one. Requires **Python 3.10+**.

---

## Quickstart

Run a one-shot query:

```bash
faraday "Summarize the main findings in my uploaded notes."
```

Interactive mode (no positional argument):

```bash
faraday
```

Optional runtime config (see `faraday.example.yaml`). By default the CLI looks for `./faraday.yaml`, `FARADAY_CONFIG`, or common container paths such as `/app/config/faraday.yaml`:

```bash
export FARADAY_CONFIG=/path/to/faraday.yaml
faraday --config /path/to/faraday.yaml "Your task here"
```

Useful flags: `--model`, `--max-steps`, `--debug`, `--check-tools`. See `faraday --help`.

You will need API keys and credentials for the models and services your deployment uses (e.g. via environment variables). Use `faraday --check-tools` to verify tool availability after configuration.

---

## Docker

Faraday can run directly inside a Docker container using a YAML config file plus environment variables for secrets. For most local and self-hosted deployments, set `app.mode: host` and `sandbox.backend: docker` to run code in the `faraday-code-sandbox` container instead of Modal. For app-container deployments, set `app.mode: docker` so Docker code execution runs in sidecar mode.

### 1. Create a config file

Start with a `faraday.yaml` like this:

```yaml
name: faraday-local

backends:
  db: sqlite
  rag: in-memory

app:
  mode: docker
  workspace:
    source_root: /workspace
    # Optional: copy workspace into an isolated temp directory per run.
    # init_mode: copy
    # copy_root: /workspace/.faraday_runtime/workspace-copies
    # keep_copy: false

sandbox:
  backend: docker
  workspace:
    container_path: /workspace

outputs:
  root: /workspace/run_outputs

persistence:
  db_messages: true
```

Notes:

- `db: sqlite` is the easiest local default.
- `rag: in-memory` avoids requiring an external vector store.
- `app.mode: docker` + `sandbox.backend: docker` is the recommended app-container sidecar default.
- Set `app.workspace.init_mode: copy` when you want repeatable runs from the same
  starting files without mutating the source workspace.
- Keep API keys and credentials in environment variables, not in the YAML file.

### 2. Build the images

```bash
docker build -f Dockerfile.main -t faraday-oss .
docker build -f Dockerfile.sandbox -t faraday-code-sandbox .
```

The sandbox image includes a LaTeX toolchain so report-generation tasks can write `.tex`
sources and render polished PDFs inside the code-execution environment.

### 3. Run Faraday against a mounted workspace

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/faraday.yaml":/app/config/faraday.yaml:ro \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  faraday-oss "Summarize the files in this workspace."
```

This image defaults to:

- `FARADAY_CONFIG=/app/config/faraday.yaml`
- YAML `app.mode: docker` and `sandbox.backend: docker`
- YAML `app.workspace.source_root: /workspace`
- YAML `sandbox.workspace.container_path: /workspace`
- YAML `sandbox.docker_image: faraday-code-sandbox`
- Sidecar Docker execution fails closed if the Docker socket or `faraday-code-sandbox`
  image is missing; it does not fall back to running code inside the app container.

### 4. Optional overrides

If you want to point at a different config path or workspace explicitly:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/custom-config.yaml":/tmp/custom-config.yaml:ro \
  -e FARADAY_CONFIG=/tmp/custom-config.yaml \
  faraday-oss --workspace-source /workspace --execution-workspace-path /workspace "Your task here"
```

### 5. Typical secrets

Depending on which features and backends you enable, you may need environment variables such as:

- `OPENAI_API_KEY`
- Modal credentials if you switch `sandbox.backend` back to `modal`

### 6. Interactive mode

You can also run the container without a one-shot prompt:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD/faraday.yaml":/app/config/faraday.yaml:ro \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  faraday-oss
```

### 7. Troubleshooting

- Run `faraday --check-tools` inside the container to confirm keys and tool availability.
- If you want the Docker sidecar for code execution, keep `app.mode: docker` and `sandbox.backend: docker`.
- If docker-based code execution is enabled, mount `/var/run/docker.sock` and prebuild
  `faraday-code-sandbox` on the Docker host before launching `faraday-oss`.
- If you want the older Modal-backed execution model, set `sandbox.backend: modal` and provide the required Modal credentials.

---

## Harbor Integration

Faraday includes a Harbor installed-agent adapter at `faraday.integrations.harbor.agent:FaradayHarborAgent`. This follows Harbor's installed-agent model, where the agent is installed into the task container and run headlessly with its own tools, rather than treated as a black-box external CLI. See the [Harbor Agents docs](https://harborframework.com/docs/agents) for the underlying model.

### Why this integration shape

- It reuses `FaradayAgent` directly instead of re-implementing prompts or tool wiring.
- It keeps Docker and Harbor on the same YAML-based runtime contract.
- It uses local workspace execution by default so Harbor benchmarks the real task filesystem.

### Harbor usage

Run Faraday in Harbor with:

```bash
harbor run -d "<dataset@version>" \
  --agent-import-path faraday.integrations.harbor.agent:FaradayHarborAgent
```

### Config behavior under Harbor

- Harbor uses the same `faraday.yaml` model as Docker.
- The recommended Harbor default is `app.mode: host` with `sandbox.backend: docker`.
- That means `execute_python_code`, `execute_bash_code`, and related actions run inside the task container workspace rather than a separate Modal sandbox.

### Recommended workflow

1. Validate your `faraday.yaml` locally with Docker.
2. Keep the same config shape when moving into Harbor.
3. Benchmark with Harbor using the Faraday import path above.

---

## More information

- **Product & vision:** [ascentbio.ai](https://ascentbio.ai/)
- **Blog:** [ascentbio.ai/blog](https://ascentbio.ai/blog)
- **Social:** [X / AscentBio](https://x.com/AscentBio) · [LinkedIn](https://www.linkedin.com/company/ascent-bio/)
- **Terms & privacy:** [Terms of Use](https://ascentbio.ai/terms-of-use) · [Privacy Policy](https://ascentbio.ai/privacy)

Faraday is built by [Ascent Bio](https://ascentbio.ai/) to help scientists work at the speed of thought while keeping **your data and ideas protected by design**.
