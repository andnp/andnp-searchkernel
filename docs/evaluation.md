# Retrieval evaluation contracts

SearchKernel evaluation treats each result ID as a ranked item. The ranked
sequence is evaluated at the requested cutoff `k`; duplicate IDs remain
visible in the report and are listed in `duplicate_result_ids` so a backend
cannot silently claim unique coverage.

- `recall@k` is the fraction of labeled relevant IDs present in the first `k`
  unique results.
- `nDCG@k` uses binary relevance unless the corpus supplies graded gains. Its
  ideal ranking is built only from IDs labeled relevant in that query.
- `MRR` uses the first relevant result in the ranked sequence.
- Average precision averages precision at each relevant hit over the labeled
  relevant set.

Empty relevant sets produce zero-valued metrics. Negative cutoffs are invalid.
Evaluation reports preserve per-query metrics, latency, source coverage, and
duplicate result IDs for later quality and performance gates.
