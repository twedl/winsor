"""Benchmark-driven winsorization for grouped polars data: a port of funcs.R.

Each group's values are winsorized (the most extreme ones clamped to their
nearest remaining neighbour) so that the group's sum lands as close as
possible to the group's benchmark total.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def winsorize_n_bm(
    v: str | pl.Expr,
    bm: str | float | pl.Expr,
    bottom: int,
    top: int,
    scale: float | pl.Expr | None = None,
) -> pl.Expr:
    """Clamp the `bottom` smallest and `top` largest values of `v`.

    Port of R ``winsorize_n_bm``. Returns an expression over the whole column;
    call ``.over(group)`` on it to winsorize each group separately.

    A clamped value only counts as an outlier if ``abs(value) >= scale * bm``,
    where `scale` defaults to ``len(v) ** -0.25``. If any clamped value is not
    an outlier the whole group becomes null, as R returns ``NA``. If `bm` is
    null or ``bottom == top == 0``, `v` comes back unchanged.

    Null and NaN values are skipped when ranking and pass through untouched.
    Groups with no more than ``max(bottom, top)`` other values, where R
    raises an error, become null.
    """
    if bottom < 0 or top < 0:
        raise ValueError("bottom and top must be non-negative")
    v = _float(v)
    if bottom == 0 and top == 0:
        return v
    bm = _float(bm).first()
    ranked = v.drop_nulls().drop_nans()  # what R's sort() keeps
    if scale is None:
        scale = v.len().cast(pl.Float64).pow(-0.25)
    threshold = scale * bm

    not_outlier = pl.lit(False)
    if bottom > 0:
        not_outlier = not_outlier | (ranked.bottom_k(bottom).max().abs() < threshold)
    if top > 0:
        not_outlier = not_outlier | (ranked.top_k(top).min().abs() < threshold)

    upper = ranked.top_k(top + 1).min()
    # R clamps from below and then from above, so when the bounds cross
    # (bottom + top == n) every value ends up at the upper one.
    lower = pl.min_horizontal(ranked.bottom_k(bottom + 1).max(), upper)

    return (
        pl.when(ranked.len() <= max(bottom, top))
        .then(None)
        .when(bm.is_null() | bm.is_nan())
        .then(v)
        .when(not_outlier)
        .then(None)
        .otherwise(v.clip(lower, upper))
    )


def optimal_w_params(
    lf: pl.LazyFrame,
    value: str,
    benchmark: str,
    by: str | Sequence[str],
    *,
    max_trim: int = 10,
) -> pl.LazyFrame:
    """Choose, per group, how many values to winsorize at each end.

    Port of the search in R ``optimal_w``. Every ``(bottom, top)`` pair up to
    ``min(n // 2, max_trim)`` is scored by the distance between the
    winsorized group sum and the benchmark (the group's first `benchmark`
    value); the closest wins, with ties going to the smallest bottom, then
    top. Pairs that would clamp a non-outlier (see `winsorize_n_bm`) are
    not eligible, and ``(0, 0)`` always is.

    Instead of winsorizing every group once per pair, each pair's sum is
    worked out from the group total and its ``max_trim + 1`` smallest and
    largest values, so the cost is one pass over the data plus
    ``(max_trim + 1) ** 2`` small candidate rows per group.

    Groups with any null or NaN `value` are left unchanged: in R these make
    every candidate sum NA, so ``(0, 0)`` wins (or R errors).

    Returns one row per group: the `by` columns, ``bottom``, ``top`` and the
    clip bounds ``lower`` and ``upper``, which are null for groups that are
    left unchanged.
    """
    if max_trim < 0:
        raise ValueError("max_trim must be non-negative")
    by = _as_list(by)
    k = max_trim + 1
    v = pl.col(value).cast(pl.Float64)
    # element i of a _sum_* list: the sum of the i values before it
    sums_before = pl.element().cum_sum().shift(1, fill_value=0.0)
    stats = (
        lf.group_by(by)
        .agg(
            _n=pl.len(),
            _clean=~(v.is_null() | v.is_nan()).any(),
            _sum=v.sum(),
            _bm=pl.col(benchmark).cast(pl.Float64).first(),
            _smallest=v.bottom_k(k).sort(),
            _largest=v.top_k(k).sort(descending=True),
        )
        .with_columns(
            _sum_smallest=pl.col("_smallest").list.eval(sums_before),
            _sum_largest=pl.col("_largest").list.eval(sums_before),
        )
    )
    trims = pl.LazyFrame({"bottom": range(k)}).join(
        pl.LazyFrame({"top": range(k)}), how="cross"
    )

    n = pl.col("_n")
    bottom, top = pl.col("bottom"), pl.col("top")
    lower, upper = pl.col("lower"), pl.col("upper")
    smallest, largest = pl.col("_smallest").list, pl.col("_largest").list
    threshold = n.cast(pl.Float64).pow(-0.25) * pl.col("_bm")
    not_outlier = ((bottom > 0) & (smallest.get(bottom - 1).abs() < threshold)) | (
        (top > 0) & (largest.get(top - 1).abs() < threshold)
    )
    # Clamping adds (lower - x) for each of the `bottom` smallest values and
    # (upper - x) for each of the `top` largest.
    clamped_sum = (
        pl.col("_sum")
        + (bottom * lower - pl.col("_sum_smallest").list.get(bottom))
        + (top * upper - pl.col("_sum_largest").list.get(top))
    )

    candidates = (
        stats.join(trims, how="cross")
        .filter(bottom <= n // 2, top <= n // 2)
        .with_columns(upper=largest.get(top))
        # Bounds cross only when bottom + top == n; then R leaves every
        # value at `upper`.
        .with_columns(lower=pl.min_horizontal(smallest.get(bottom), upper))
        .with_columns(
            _score=pl.when(pl.col("_clean") & ~not_outlier)
            .then((clamped_sum - pl.col("_bm")).abs())
            .fill_nan(None)
        )
        # Pairs with the same bounds produce the same values, so in R their
        # sums tie exactly; share one score so rounding here can't split them.
        .with_columns(
            _score=pl.when(pl.col("_score").is_not_null()).then(
                pl.col("_score").min().over([*by, "lower", "upper"])
            )
        )
    )
    unchanged = (bottom == 0) & (top == 0)
    return (
        candidates.group_by(by)
        .agg(
            pl.col("bottom", "top", "lower", "upper")
            .sort_by("_score", "bottom", "top", nulls_last=True)
            .first()
        )
        .with_columns(
            lower=pl.when(~unchanged).then(lower),
            upper=pl.when(~unchanged).then(upper),
        )
    )


def optimal_w(
    lf: pl.LazyFrame,
    value: str,
    benchmark: str,
    by: str | Sequence[str],
    *,
    out: str | None = None,
    max_trim: int = 10,
) -> pl.LazyFrame:
    """Winsorize `value` within each group so its sum best matches `benchmark`.

    Port of R ``optimal_w``, run on every group of `by` at once (see
    `optimal_w_params` for how each group's winsorization is chosen). Adds
    the Float64 column `out`, by default ``f"{value}_w"``, and leaves the
    rest of `lf`, including row order, as it was.
    """
    by = _as_list(by)
    bounds = optimal_w_params(lf, value, benchmark, by, max_trim=max_trim).select(
        *by, _lower=pl.col("lower"), _upper=pl.col("upper")
    )
    return (
        lf.join(bounds, on=by, how="left", nulls_equal=True, maintain_order="left")
        # null bounds clip nothing, leaving unchanged groups as they were
        .with_columns(
            pl.col(value)
            .cast(pl.Float64)
            .clip(pl.col("_lower"), pl.col("_upper"))
            .alias(out or f"{value}_w")
        )
        .drop("_lower", "_upper")
    )


def _float(x: str | float | pl.Expr) -> pl.Expr:
    if isinstance(x, str):
        x = pl.col(x)
    elif not isinstance(x, pl.Expr):
        x = pl.lit(x)
    return x.cast(pl.Float64)


def _as_list(by: str | Sequence[str]) -> list[str]:
    return [by] if isinstance(by, str) else list(by)
