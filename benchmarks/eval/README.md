# Eval harness

Dataset, Model, Runner, Logger. Which set to run: [`docs/benchmarks.md`](../../docs/benchmarks.md). Membership: [`sets.py`](sets.py).

```bash
python -m benchmarks.eval --list-sets
make eval-smoke
```

```python
from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset


@dataset("my_dataset")
class MyDataset(Dataset): ...
```

Import the module from the package `__init__.py`. Add the name to `SETS` in `sets.py` to put it on a named set.
