# Eval Harness

Clean benchmark harness for recursive-flow. The core components are:

- `Dataset` - examples and scoring.
- `Model` - inference.
- `Runner` - execution strategy.
- `Logger` - side effects.

Initial datasets:

- `synthetic_needle`
- `oolong`

## Smoke

```bash
make eval-smoke
```

Direct CLI:

```bash
python -m benchmarks.eval \
  --model fake \
  --dataset synthetic_needle \
  --runner fake vanilla rflow-local \
  --seeds 0:3 \
  --dataset-param synthetic_needle.records=8 \
  --dataset-param synthetic_needle.filler_words=2
```

## Real Run

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset oolong \
  --runner vanilla rflow-local official-rlm \
  --seeds 0:20
```

Rows and artifacts are written under `benchmarks/runs/<run_id>/`.

## Adding Components

Datasets:

```python
from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset


@dataset("my_dataset")
class MyDataset(Dataset):
    ...
```

Runners:

```python
from benchmarks.eval import runner
from benchmarks.eval.types import Runner


@runner("my-runner")
class MyRunner(Runner):
    ...
```

Decorators are only for component registration. Built-ins are imported
explicitly from package `__init__.py` files.
