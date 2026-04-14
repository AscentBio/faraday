# Faraday Agent

[![PyPI version](https://img.shields.io/pypi/v/faradayai.svg)](https://pypi.org/project/faradayai/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

### Accelerating science with agentic AI for research and drug design


[Faraday](https://ascentbio.ai/) is an AI scientist that carries out research workflows end to end:


## Install

```bash
pip install faradayai
```

---

## Credentials

Set a model API key before running. Faraday supports OpenAI, OpenRouter, and Azure OpenAI.

### OpenAI (default)

```bash
export OPENAI_API_KEY=...
```

### OpenRouter

```bash
export OPENROUTER_API_KEY=...
```

```yaml
# faraday.yaml
llm:
  provider: openrouter
  model: openai/gpt-5
  api_key_env: OPENROUTER_API_KEY
```

### Azure OpenAI

```yaml
# faraday.yaml
llm:
  provider: azure
  model: gpt-5
  api_key: OPENAI_API_KEY
  base_url: AZURE_OPENAI_BASE_URL
  api_version: preview
```

Run `faraday --check-tools` to verify keys and tool availability after setup.

---

## Quickstart

Run a one-shot task:

```bash
faraday "Summarize structure–activity relationships across KRAS inhibitors, focusing on motifs that improve binding affinity and selectivity"
```

Interactive mode:

```bash
faraday
```

With a config file:

```bash
faraday --config faraday.example.yaml "Your task here"
```

Results are written to `./run_outputs/` by default:

```
run_outputs/
  run_{timestamp}_{chat_id}_{query_id}/
    agent_outputs/       # files, plots, and generated artifacts
    run_artifacts/
      events.jsonl       # timestamped stream of every agent event
      metadata.json      # run parameters (model, steps, config)
      result.json        # final answer and run summary
      trajectory.json    # full trajectory for replay
```

Useful flags: `--model`, `--max-steps`, `--debug`, `--check-tools`, `--app-mode`, `--sandbox-backend`, `--batch-file`, `--use-docker`. See `faraday --help` for the full list.

