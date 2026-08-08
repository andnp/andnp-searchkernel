# SearchKernel release-readiness evidence

This checklist separates deterministic correctness evidence from observations
that depend on the host running the benchmark. It is required before claiming
production readiness for a search rollout.

## Automated evidence

- [ ] The safe CI suite passes with locked dependencies, Ruff, Pyrefly,
  import-linter, strict markers, and the coverage floor.
- [ ] The readiness job produces a labeled report with at least two warmup
  calls and five measured repetitions per query.
- [ ] The readiness artifact passes schema validation and the configured
  baseline quality gate. The gate checks recall, nDCG, MRR, and AP; its
  thresholds live in `benchmarks/evidence-policy.json`.
- [ ] The benchmark job publishes serial/concurrent latency observations and
  synthetic 1k/10k/100k resource evidence. These are evidence artifacts, not
  absolute latency promises.

## Production evidence to attach

- [ ] Corpus version, query-label source, backend, model fingerprint, vector
  dimension, filters, configuration fingerprint, and environment are recorded.
- [ ] Cold-start, warm-cache, serial, and representative-concurrency runs are
  attached, with p50/p95/p99 latency, QPS, RSS, index size, and build/load
  times where applicable.
- [ ] Quality is reported by query type, source kind, workspace, and important
  failure or empty-result slices.
- [ ] Local/Postgres/FAISS parity evidence covers the enabled deployment path;
  unavailable optional dependencies are recorded as gaps rather than passes.
- [ ] Capacity, degradation, timeout, restart, and rollback observations are
  linked to an owner and a collection date.

## Decision record

The release owner records the artifact names, the baseline and policy used,
known gaps, mitigations, rollback trigger, and approval date. Do not convert a
machine-sensitive latency observation into an absolute CI threshold. If a
latency gate is needed for a controlled environment, enable the relative gate
explicitly in the policy and document why that environment is stable.
