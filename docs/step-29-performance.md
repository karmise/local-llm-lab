# Bounded performance and load checks

A small closed-loop Python harness measures non-streaming end-to-end HTTP latency, completed requests per second and failure rate. It supports one to four simultaneous workers and a total budget of one to twenty requests. Every RAG request owns a separate HTTP session; document setup/indexing is outside the timing window. The fixture cleans up the shared temporary workspace after all workers finish.

From `automation`, start with a cheap concurrency check:

```bash
.venv/bin/python -m pytest tests/test_performance.py --run-performance \
  --performance-mode health --performance-requests 6 --performance-users 2 \
  --performance-p95 5 --alluredir reports/performance/allure-results
```

One model request, explicitly enabled:

```bash
.venv/bin/python -m pytest tests/test_performance.py --run-performance \
  --performance-mode rag --rag-model qwen3.5:4b \
  --performance-requests 1 --performance-users 1 --performance-p95 180
```

RAG requests also check the carryover golden facts and sources: a fast but wrong answer is a failed request. Timeout, HTTP and acceptance failures remain in the attempt evidence; no retries occur. JSON reports include every attempt, p95 using nearest rank, median, model digest, configuration, workload, machine/Python information and declared thresholds. They are saved before assertions, so failed runs retain evidence. The default workload is health; model calls require selecting `rag`. The single-answer capture mode is unsupported for this batch workload.

The p95 limit is a declared experiment criterion, not a calibrated service SLO. At very small sample counts it is effectively an observed maximum. Timing includes connection establishment, response transfer and validation, and may include a cold model load; warmup is explicitly recorded as zero. This is not streaming TTFT, server inference duration, a sustained-arrival-rate benchmark or proof of production capacity. Use a controlled host, comparable warmup and larger representative runs before drawing such conclusions. For large distributed workloads, a dedicated tool such as [Locust](https://docs.locust.io/en/stable/writing-a-locustfile.html) can be added separately; it is not a dependency of this bounded harness.

Retained comparable batches can also detect performance regression:

```bash
.venv/bin/python -m llm_testkit.performance.comparison reports/performance/<baseline>.json \
  reports/performance/<current>.json --maximum-growth 0.2 --output reports/performance/comparison.json
```

This explicitly chosen baseline must be healthy. Missing or changed workload, worker count, request count, machine, Python, policy or non-model configuration yields an incomparable result. Failed current requests or a p95 increase greater than the declared fraction yield regression and a nonzero exit. Model digest changes are disclosed. Host metadata does not measure background load or guarantee identical hardware state; small batches are exploratory.
