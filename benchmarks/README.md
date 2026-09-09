# benchmarks/

Named eval sets. Source of truth: [`docs/benchmarks.md`](../docs/benchmarks.md). Membership: [`eval/sets.py`](eval/sets.py).

```bash
python -m benchmarks.eval --list-sets
make eval-smoke
make eval-reasoning EVAL_MODEL=gpt-5-mini
```
