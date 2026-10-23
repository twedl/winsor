"""Benchmark-driven winsorization for grouped polars data.

Each group's values are winsorized (the most extreme ones clamped to their
nearest remaining neighbour) so that the group's sum lands as close as
possible to the group's benchmark total. Values and benchmarks are
assumed to be Float64.
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

    Clamped values are set to the nearest value that's kept. This returns an
    expression, so use ``.over(group)`` to winsorize each group separately.

    Only outliers get clamped. A value counts as an outlier if it's a big
    chunk of the benchmark total on its own (``abs(value) >= scale * bm``).
    By default ``scale = len(v) ** -0.25`` with nulls included, which works
    out to 32% of the benchmark for 100 values, or 10% for 10,000. At each
    end we only check the least extreme of the values being clamped.

    The result is:

    - `v` unchanged if ``bottom == top == 0``, or if `bm` is null or NaN.
    - All nulls if the clamp is rejected, either because it wouldn't leave
      any values or because it would clamp a value that isn't an outlier.
    - Otherwise, the clamped values.

    If `bm` is a column, its first value is used. Nulls and NaNs in `v` are
    ignored when ranking and come back unchanged.
    """
    v = _expr(v)
    if bottom == 0 and top == 0:  # nothing to clamp
        return v
    bm = _expr(bm).first()  # if bm is a column, use its first value
    ranked = v.drop_nulls().drop_nans()  # leave nulls and NaNs out of ranking
    if scale is None:
        scale = v.len().pow(-0.25)  # len() includes nulls
    # A value is an outlier if it's a big chunk of the benchmark total on its
    # own, i.e. abs(value) >= threshold. With the default scale that's 32% of
    # the benchmark for 100 values, or 10% for 10,000. Using abs() means big
    # negative values count the same as big positive ones.
    threshold = scale * bm

    # We only clamp outliers. At each end we check the least extreme value
    # being clamped, and if it isn't an outlier we reject the clamp.
    innermost_low = ranked.bottom_k(bottom).max()
    innermost_high = ranked.top_k(top).min()
    not_outlier = ((bottom > 0) & (innermost_low.abs() < threshold)) | (
        (top > 0) & (innermost_high.abs() < threshold)
    )

    # lower and upper are the smallest and largest values we keep. Anything
    # below lower gets raised to it, and anything above upper gets lowered.
    lower = ranked.bottom_k(bottom + 1).max()
    upper = ranked.top_k(top + 1).min()
    clamped = v.clip(lower, upper)

    # With no benchmark we can't tell what's an outlier, so leave v alone.
    no_benchmark = bm.is_null() | bm.is_nan()
    # Reject the clamp if it wouldn't leave any values, or if it would clamp a
    # value that isn't an outlier.
    rejected = (ranked.len() <= bottom + top) | not_outlier

    # if no_benchmark: v, elif rejected: null, else: clamped
    return (
        pl.when(no_benchmark).then(v)
        .when(rejected).then(None)
        .otherwise(clamped)
    )


def optimal_w_params(
    lf: pl.LazyFrame,
    value: str,
    benchmark: str,
    by: str | Sequence[str],
    *,
    max_trim: int = 10,
) -> pl.LazyFrame:
    """Find the best ``(bottom, top)`` pair for each group.

    Returns one row per group with the winning pair, not the winsorized
    values or their sum. The columns are:

    - the `by` columns
    - ``bottom`` and ``top``, how many values to clamp at each end
    - ``lower`` and ``upper``, the smallest and largest values kept, which
      `optimal_w` clips to. They're null when the pair is ``(0, 0)``, which
      means leave the group alone.

    Nothing gets clamped here. `optimal_w` does the clamping, and
    `winsorize_n_bm` isn't used at all.

    Tries every ``(bottom, top)`` pair up to ``min(n // 2, max_trim)`` that
    leaves at least one value, and works out how far the winsorized sum
    would be from the benchmark (the group's first `benchmark` value). The
    pair that gets closest wins. Ties go to the smaller bottom, then the
    smaller top. ``(0, 0)``, which leaves the group alone, is always allowed.

    A pair is skipped if it would clamp a value that isn't an outlier. A
    value counts as an outlier if it's a big chunk of the benchmark total on
    its own (``abs(value) >= n ** -0.25 * benchmark``). At each end we only
    check the least extreme of the values being clamped.

    It reads `lf` twice and works with the streaming engine, so `lf` never
    has to fit in memory. The first pass gets each group's count, total and
    benchmark, plus cutoffs that the ``max_trim + 1`` smallest and largest
    values must lie beyond. The second pass keeps only the rows past those
    cutoffs, usually under 20 per group at each end. Each pair's winsorized
    sum is then worked out from those values and the group total.

    Groups with any null or NaN `value` are left unchanged. Values are
    assumed finite; the sums are not valid with ±Inf.
    """
    by = _as_list(by)
    # Up to max_trim values get clamped at each end, and we also need the
    # value they're clamped to.
    k = max_trim + 1
    v = pl.col(value)

    # First pass. bottom_k and top_k don't work with the streaming engine,
    # so we can't just take each group's k smallest and largest values.
    # Instead we deal each group's values into piles (by hashing the value)
    # and take each pile's min and max, which do stream. If a group has at
    # least k piles, at least k of its values are at or below the k-th
    # smallest pile min, so its k smallest values are too. Same for the top.
    piles = max(16, k + 5)
    pile = v.hash() % piles
    pile_mins = [f"_pile_min{i}" for i in range(piles)]
    pile_maxes = [f"_pile_max{i}" for i in range(piles)]
    pile_stats = [v.filter(pile == i).min().alias(c) for i, c in enumerate(pile_mins)]
    pile_stats += [v.filter(pile == i).max().alias(c) for i, c in enumerate(pile_maxes)]
    stats = (
        lf.group_by(by)
        .agg(
            *pile_stats,
            _n=pl.len(),  # includes nulls
            _clean=~(v.is_null() | v.is_nan()).any(),  # no nulls or NaNs
            _sum=v.sum(),
            _bm=pl.col(benchmark).first(),
        )
        # With fewer than k piles there's no cutoff, so we keep every row.
        .with_columns(
            _low_cut=pl.concat_list(pile_mins)
            .list.drop_nulls()
            .list.sort()
            .list.get(k - 1, null_on_oob=True)
            .fill_null(float("inf")),
            _high_cut=pl.concat_list(pile_maxes)
            .list.drop_nulls()
            .list.sort(descending=True)
            .list.get(k - 1, null_on_oob=True)
            .fill_null(float("-inf")),
        )
        .select(*by, "_n", "_clean", "_sum", "_bm", "_low_cut", "_high_cut")
    )

    # Second pass. Only rows past the cutoffs can be among a group's k
    # smallest or largest values, and there are only a few of them, so
    # bottom_k and top_k on what's left is cheap.
    extremes = (
        lf.join(stats.select(*by, "_low_cut", "_high_cut"), on=by, nulls_equal=True)
        .filter((v <= pl.col("_low_cut")) | (v >= pl.col("_high_cut")))
        .group_by(by)
        .agg(
            _smallest=v.bottom_k(k).sort(),  # ascending
            _largest=v.top_k(k).sort(descending=True),  # descending
        )
    )
    stats = stats.join(extremes, on=by, how="left", nulls_equal=True)

    # A value is an outlier if it's a big chunk of the benchmark total on its
    # own, i.e. abs(value) >= threshold.
    n = pl.col("_n")
    threshold = n.pow(-0.25) * pl.col("_bm")
    # Running sum shifted by one, so element i is the sum of the first i
    # values. e.g. [a, b, c] -> [0, a, a + b]
    sums_before = pl.element().cum_sum().shift(1, fill_value=0.0)

    def end(values: str, trim: str, bound: str, clamped: str) -> pl.LazyFrame:
        # One row per group and number of values to clamp at one end (trim),
        # with the value they get clamped to (bound) and their sum (clamped).
        # e.g. smallest values [-100, 1, 2] give trims 0, 1, 2, bounds -100,
        # 1, 2 and clamped sums 0, -100, -99.
        return (
            stats.select(
                *by,
                "_n",
                "_clean",
                "_bm",
                pl.int_ranges(pl.col(values).list.len()).alias(trim),
                pl.col(values).alias(bound),
                # the least extreme value being clamped
                pl.col(values).list.eval(pl.element().shift(1)).alias("_check"),
                pl.col(values).list.eval(sums_before).alias(clamped),
            )
            .explode(trim, bound, "_check", clamped)
            # a group with only nulls has no values, but still gets trim 0
            .with_columns(pl.col(trim).fill_null(0))
            # Clamp at most half the values, and only if the least extreme
            # one is an outlier. Trim 0 clamps nothing, so it's always kept.
            # Groups with nulls or NaNs, or no benchmark, only keep trim 0.
            .filter(
                pl.col(trim) <= pl.col("_n") // 2,
                (pl.col(trim) == 0)
                | (pl.col("_clean") & (pl.col("_check").abs() >= threshold)),
            )
            .select(*by, trim, bound, clamped)
        )

    bottom, top = pl.col("bottom"), pl.col("top")
    lower, upper = pl.col("lower"), pl.col("upper")
    # The sum we'd get after clamping, without clamping anything. The bottom
    # smallest values each go up to lower and the top largest each come down
    # to upper, so we add those differences to the total. e.g. [-100, 1, 2, 3,
    # 50] with bottom = top = 1 is -44 + (1 + 100) + (3 - 50) = 10, which is
    # the sum of [1, 1, 2, 3, 3].
    clamped_sum = (
        pl.col("_sum")
        + (bottom * lower - pl.col("_clamped_low"))
        + (top * upper - pl.col("_clamped_high"))
    )

    # Every combination of a bottom trim and a top trim for each group. Most
    # groups have no outliers, so they only get (0, 0).
    candidates = (
        end("_smallest", "bottom", "lower", "_clamped_low")
        .join(end("_largest", "top", "upper", "_clamped_high"), on=by, nulls_equal=True)
        .join(stats.select(*by, "_n", "_sum", "_bm"), on=by, nulls_equal=True)
        .filter(bottom + top < pl.col("_n"))  # leave at least one value
        # The score is how far the clamped sum is from the benchmark, and
        # lower is better. With no benchmark it's null (fill_nan turns a NaN
        # benchmark's scores into nulls too), and (0, 0) wins the tie.
        .with_columns(_score=(clamped_sum - pl.col("_bm")).abs().fill_nan(None))
        # Two pairs with the same lower and upper clamp to exactly the same
        # values, so they should tie. Rounding in clamped_sum can make their
        # scores differ a tiny bit, so give them all the smallest one.
        .with_columns(_score=pl.col("_score").min().over([*by, "lower", "upper"]))
    )

    # For each group, take the pair with the lowest score. Ties go to the
    # smaller bottom, then the smaller top. (0, 0) is always a candidate, so
    # every group gets a pair.
    unchanged = (bottom == 0) & (top == 0)
    return (
        candidates.group_by(by)
        .agg(
            pl.col("bottom", "top", "lower", "upper")
            .sort_by("_score", "bottom", "top", nulls_last=True)
            .first()
        )
        # (0, 0) means leave the group alone. Null bounds make optimal_w
        # skip the clip.
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

    Picks the best ``(bottom, top)`` pair for each group with
    `optimal_w_params`, then clips each group's values to that pair's
    bounds. Adds the column `out` (``f"{value}_w"`` by default) and leaves
    everything else in `lf`, including row order, as it was.

    `lf` gets read three times, twice to pick the pairs and once to clip. If
    it's an expensive query rather than a file scan, sink it to a file
    first. With hundreds of thousands of groups, polars' streaming group_by
    still needs a lot of memory, and running one year (or other `by` value)
    at a time keeps it down.
    """
    by = _as_list(by)
    # The best pair for each group. We only need the bounds to clip, and we
    # rename them so they can't clash with a column already in lf.
    bounds = optimal_w_params(lf, value, benchmark, by, max_trim=max_trim).select(
        *by, _lower=pl.col("lower"), _upper=pl.col("upper")
    )
    return (
        # Give every row its group's bounds. A left join keeps every row in
        # its original order, and nulls_equal lets null group keys match.
        lf.join(bounds, on=by, how="left", nulls_equal=True, maintain_order="left")
        # Null bounds don't clip anything, so groups we left alone come back
        # unchanged.
        .with_columns(
            pl.col(value)
            .clip(pl.col("_lower"), pl.col("_upper"))
            .alias(out or f"{value}_w")
        )
        .drop("_lower", "_upper")
    )


def _expr(x: str | float | pl.Expr) -> pl.Expr:
    if isinstance(x, str):
        return pl.col(x)
    return x if isinstance(x, pl.Expr) else pl.lit(x)


def _as_list(by: str | Sequence[str]) -> list[str]:
    return [by] if isinstance(by, str) else list(by)
