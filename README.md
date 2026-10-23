# winsor

Winsorize values within groups so each group's sum is close to its benchmark.

## Install

```sh
uv sync
```

## Data

Hive-partitioned parquet files, like

```
data/year=2022/prov=10/0.parquet
data/year=2022/prov=11/0.parquet
data/year=2023/prov=10/0.parquet
...
```

with columns `ioic`, `prm`, `value` and `benchmark`.

## Winsorize all of it

```python
import polars as pl
from winsor import optimal_w

lf = pl.scan_parquet("data/", hive_partitioning=True)
by = ["year", "prov", "ioic", "prm"]

optimal_w(lf, "value", "benchmark", by=by).sink_parquet(
    pl.PartitionBy("winsorized/", key=["year", "prov"])
)
```

The output has the same columns plus `value_w`, the winsorized values.

## One year at a time

```python
for year in [2022, 2023]:
    one_year = lf.filter(pl.col("year") == year)
    optimal_w(one_year, "value", "benchmark", by=by).sink_parquet(
        pl.PartitionBy("winsorized/", key=["year", "prov"])
    )
```

## See what was clamped

```python
from winsor import optimal_w_params

optimal_w_params(lf, "value", "benchmark", by=by).sink_parquet("clamps.parquet")
```

This has one row per group. `bottom` and `top` are how many values were
clamped at each end, and `lower` and `upper` are the values they were clamped
to. `lower` and `upper` are null if nothing was clamped.

## Tests

```sh
uv run pytest
```
