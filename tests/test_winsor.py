import math
import random
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from winsor import optimal_w, optimal_w_params, winsorize_n_bm

### some test parameters (tests.R)

L = 100
_rng = random.Random(1)
x = [_rng.gauss(1, 1) for _ in range(L)]
bm = float(L)


def winsorize(v, bm, bottom, top, scale=None):
    """Single-group winsorize_n_bm; None where R returns NA."""
    out = pl.select(
        winsorize_n_bm(pl.Series("v", v, dtype=pl.Float64), bm, bottom, top, scale)
    ).to_series()
    return None if out.is_null().all() else out.to_list()


def opt_w(value, benchmark):
    """Single-group optimal_w."""
    lf = pl.LazyFrame(
        {"g": 1, "value": pl.Series(value, dtype=pl.Float64), "benchmark": benchmark}
    )
    return optimal_w(lf, "value", "benchmark", "g").collect()["value_w"].to_list()


def test_fix_top_1():
    assert winsorize([*x, 100], bm, bottom=0, top=1) == [*x, max(x)]


def test_no_outliers_top_returns_na():
    assert winsorize(x, bm, bottom=0, top=1) is None


def test_no_outliers_bottom_returns_na():
    assert winsorize(x, bm, bottom=1, top=0) is None


def test_no_outliers_both_returns_na():
    assert winsorize(x, bm, bottom=1, top=1) is None


def test_fix_bottom_1():
    assert winsorize([-100, *x], bm, bottom=1, top=0) == [min(x), *x]


def test_fix_bottom_1_top_1():
    assert winsorize([-100, *x, 100], bm, bottom=1, top=1) == [min(x), *x, max(x)]


def test_false_negative_outliers_scale_high():
    assert winsorize([-100, *x], bm, bottom=1, top=0, scale=L) is None


def test_false_positive_outliers_scale_low():
    y = [sorted(x)[1] if v == min(x) else v for v in x]
    assert winsorize(x, bm, bottom=1, top=0, scale=0.001) == y


# test that optimal_w will find the correct winsorization parameters


def test_optimal_w_no_winsorization():
    assert opt_w(x, bm) == x


@pytest.mark.xfail(
    reason="as in tests.R: the two outliers offset, so the unwinsorized sum "
    "can be just as close to the benchmark"
)
def test_optimal_w_winsorizes_both():
    assert opt_w([-100, *x, 100], bm) == [min(x), *x, max(x)]


def test_optimal_w_winsorizes_bottom():
    assert opt_w([-100, *x], bm) == [min(x), *x]


### beyond tests.R


def ref_winsorize(v, bm, bottom, top, scale=None):
    """Port of R winsorize_n_bm (no NA values); None for NA. Unlike R,
    clamping every value (bottom + top >= n) is rejected."""
    if scale is None:
        scale = len(v) ** -0.25
    if bm is None or (bottom == 0 and top == 0):
        return list(v)
    s = sorted(v)
    if bottom + top >= len(s):
        return None
    bottom_value, top_value = s[bottom], s[len(s) - 1 - top]
    if bottom > 0 and abs(s[bottom - 1]) < scale * bm:
        return None
    if top > 0 and abs(s[len(s) - top]) < scale * bm:
        return None
    return [min(max(e, bottom_value), top_value) for e in v]


def ref_optimal_w(value, benchmark):
    """Line-by-line port of R optimal_w; also returns the chosen (i, j)."""
    try_i = range(min(len(value) // 2, 10) + 1)
    best = (math.inf, 0, 0)  # NA sums sort last, so (0, 0) if nothing else
    for i in try_i:
        for j in try_i:
            w = ref_winsorize(value, benchmark, i, j)
            if w is not None and benchmark is not None:
                best = min(best, (abs(math.fsum(w) - benchmark), i, j))
    _, i, j = best
    return ref_winsorize(value, benchmark, i, j), (i, j)


def random_groups(seed, n_groups, step):
    """Groups of 1..60 values on a grid of `step` (sums are exact, so ties
    break the same way everywhere) with outliers, duplicates and odd
    benchmarks mixed in."""
    rng = random.Random(seed)
    rows = []
    for g in range(n_groups):
        n = rng.choice([1, 2, 3, 4, 5, 6, 7, 10, 11, 20, 21, 22, 23, 40, 60])
        vals = [rng.gauss(rng.uniform(-3, 5), rng.uniform(0.1, 3)) for _ in range(n)]
        for _ in range(rng.choice([0, 0, 1, 2, 3, 12])):
            vals[rng.randrange(n)] = rng.choice([-1, 1]) * rng.uniform(20, 400)
        if rng.random() < 0.3:  # heavy ties
            vals = [round(e) for e in vals]
        vals = [float(round(e / step) * step) for e in vals]
        bench = rng.choice(
            [None, 0.0, -50.0, float(round(sum(vals) + rng.gauss(0, 30)))]
            + [float(round(rng.uniform(0, 3) * n))] * 4
        )
        rows += [(g, e, bench) for e in vals]
    return pl.DataFrame(rows, schema=["g", "value", "bm"], orient="row")


@pytest.mark.parametrize("step", [1, 0.25, 2**-20])
def test_optimal_w_matches_reference(step):
    assert_matches_reference(random_groups(seed=step, n_groups=400, step=step))


def test_optimal_w_matches_reference_large_groups():
    # Big enough groups that the first pass's cutoffs actually drop rows,
    # including groups with only a few distinct values.
    rng = random.Random(5)
    rows = []
    for g in range(30):
        n = rng.choice([200, 1000, 2000])
        if g % 3 == 0:
            vals = [float(rng.randint(0, 4)) for _ in range(n)]
        else:
            vals = [round(rng.gauss(1, 1) * 1024) / 1024 for _ in range(n)]
        for _ in range(rng.choice([0, 1, 3, 12])):
            vals[rng.randrange(n)] = float(rng.choice([-1, 1]) * rng.randint(200, 5000))
        bench = rng.choice([float(n), 0.0, float(round(sum(vals)))])
        rows += [(g, e, bench) for e in vals]
    assert_matches_reference(pl.DataFrame(rows, schema=["g", "value", "bm"], orient="row"))


def test_repeated_values_match_reference():
    # Lots of zeros at the bottom and copies of an outlier at each end. Every
    # copy of a value lands in the same pile, so they get past the cutoffs
    # together and have to be counted rather than kept row by row.
    rng = random.Random(9)
    rows = []
    for g in range(6):
        vals = [0.0] * 2000 + [round(rng.uniform(0, 2) * 1024) / 1024 for _ in range(1000)]
        vals[:g] = [-800.0] * g
        vals[-5:] = [900.0] * 5
        bm = rng.choice([float(round(sum(vals[g:-5]))), 0.0, 1000.0])
        rows += [(g, e, bm) for e in vals]
    assert_matches_reference(pl.DataFrame(rows, schema=["g", "value", "bm"], orient="row"))


def test_cutoff_keeps_just_enough_values():
    # optimal_w_params deals values into 16 piles by hash and cuts at the 11th
    # smallest pile min. Here the 10 outliers and the smallest ordinary value
    # are each the min of their own pile, so exactly 11 values make the cut,
    # and the winning lower bound is the 11th.
    def pile(x):
        return pl.Series([x]).hash()[0] % 16

    ordinary = [1 + i / 1024 for i in range(200)]
    used = {pile(ordinary[0])}
    outliers = []
    x = -1000.0
    while len(outliers) < 10:
        if pile(x) not in used:
            used.add(pile(x))
            outliers.append(x)
        x -= 1
    values = outliers + ordinary
    bm = 10 * ordinary[0] + sum(ordinary)  # what clamping all 10 outliers gives
    lf = pl.LazyFrame({"g": 0, "value": values, "bm": bm})
    params = optimal_w_params(lf, "value", "bm", "g").collect()
    assert params.select("bottom", "top", "lower").row(0) == (10, 0, ordinary[0])


def assert_matches_reference(df):
    got = optimal_w(df.lazy(), "value", "bm", "g").collect()
    params = optimal_w_params(df.lazy(), "value", "bm", "g").collect()
    picked = {g: (b, t) for g, b, t in params.select("g", "bottom", "top").iter_rows()}
    for (g,), grp in got.group_by("g", maintain_order=True):
        want, ij = ref_optimal_w(grp["value"].to_list(), grp["bm"][0])
        assert grp["value_w"].to_list() == want, f"group {g}"
        assert picked[g] == ij, f"group {g}"


@pytest.mark.parametrize("bottom", [0, 1, 2, 5])
@pytest.mark.parametrize("top", [0, 1, 3])
@pytest.mark.parametrize("scale", [None, 0.01])
def test_winsorize_matches_reference(bottom, top, scale):
    df = random_groups(seed=7, n_groups=200, step=0.25)
    got = df.with_columns(
        w=winsorize_n_bm("value", "bm", bottom, top, scale).over("g")
    )
    for (g,), grp in got.group_by("g", maintain_order=True):
        v = grp["value"].to_list()
        want = ref_winsorize(v, grp["bm"][0], bottom, top, scale)
        assert (None if grp["w"].is_null().all() else grp["w"].to_list()) == want


def test_optimal_w_agrees_with_winsorize_n_bm():
    df = random_groups(seed=3, n_groups=300, step=0.25)
    got = optimal_w(df.lazy(), "value", "bm", "g").collect()
    params = optimal_w_params(df.lazy(), "value", "bm", "g").collect()
    for g, bottom, top in params.select("g", "bottom", "top").iter_rows():
        grp = got.filter(g=g)
        w = grp.select(winsorize_n_bm("value", "bm", bottom, top))
        assert_series_equal(w.to_series(), grp["value_w"], check_names=False)


def test_equivalent_trims_tie_like_r():
    # Clamping the top 1, 2 or 3 values gives the same result since 19.99
    # repeats, so R's sums tie exactly and it reports top = 1.
    lf = pl.LazyFrame(
        {"g": 0, "value": [19.99, 0.7, 19.99, 150.7, 19.99, 0.7], "bm": 30.0}
    )
    params = optimal_w_params(lf, "value", "bm", "g").collect()
    assert params.select("bottom", "top").row(0) == (0, 1)


def test_groups_are_independent_and_order_kept():
    cases = [x, [-100, *x], [*x, 100], [5.0], [3.0, -400.0]]
    single = [opt_w(v, bm) for v in cases]
    df = pl.DataFrame(
        {"g": [f"g{i}" for i, v in enumerate(cases) for _ in v],
         "value": pl.Series([e for v in cases for e in v], dtype=pl.Float64),
         "benchmark": bm}
    ).sample(fraction=1, shuffle=True, seed=0)
    got = optimal_w(df.lazy(), "value", "benchmark", "g").collect()
    assert_frame_equal(got.drop("value_w"), df)
    for i, want in enumerate(single):
        winsorized = dict(zip(cases[i], want))  # values within a case are distinct
        grp = got.filter(g=f"g{i}")
        assert grp["value_w"].to_list() == [winsorized[e] for e in grp["value"]]


def test_multiple_group_columns_and_null_keys():
    df = pl.DataFrame(
        {"a": [1, 1, 1, None, None, None], "b": ["x"] * 6,
         "value": [1.0, 2.0, 90.0, -90.0, 1.0, 2.0], "bm": [3.0] * 6}
    )
    got = optimal_w(df.lazy(), "value", "bm", ["a", "b"], out="w").collect()
    assert got["w"].to_list() == [1.0, 2.0, 2.0, 1.0, 1.0, 2.0]


def test_missing_values_leave_group_unchanged():
    nan = float("nan")
    df = pl.DataFrame(
        {"g": [1] * 4 + [2] * 4 + [3] * 4 + [4] * 2,
         "value": [1.0, 2.0, None, 500.0, 1.0, nan, 2.0, 500.0, 1.0, 2.0, 2.0, 500.0,
                   None, None],
         "bm": [5.0] * 4 + [5.0] * 4 + [None] * 4 + [5.0] * 2}
    )
    got = optimal_w(df.lazy(), "value", "bm", "g").collect()
    assert_series_equal(got["value_w"], df["value"], check_names=False)
    params = optimal_w_params(df.lazy(), "value", "bm", "g").collect().sort("g")
    assert params.rows() == [(g, 0, 0, None, None) for g in [1, 2, 3, 4]]


def test_winsorize_skips_missing_values():
    v = pl.Series([None, -500.0, 1.0, float("nan"), 2.0, 3.0])
    got = pl.select(winsorize_n_bm(v, 5.0, bottom=1, top=0)).to_series()
    want = pl.Series([None, 1.0, 1.0, float("nan"), 2.0, 3.0])
    assert_series_equal(got, want, check_names=False)


### against the R code itself, when R is around

FUNCS_R = Path(__file__).parents[1] / "funcs.R"
R_SCRIPT = """
suppressMessages({library(dplyr); library(tidyr); library(purrr)})
args <- commandArgs(trailingOnly = TRUE)
source(args[[1]])
read.csv(args[[2]]) |>
  group_by(g) |>
  mutate(value_w = sprintf("%.17g", optimal_w(value, bm))) |>
  ungroup() |>
  write.csv(args[[3]], row.names = FALSE)
"""


def r_available():
    if shutil.which("Rscript") is None or not FUNCS_R.exists():
        return False
    load = "for (p in c('dplyr','tidyr','purrr')) library(p, character.only=TRUE)"
    ran = subprocess.run(["Rscript", "-e", load], check=False, capture_output=True)
    return ran.returncode == 0


@pytest.mark.skipif(not r_available(), reason="needs Rscript with dplyr, tidyr, purrr")
@pytest.mark.parametrize("step", [1, 2**-10])
def test_optimal_w_matches_r(tmp_path, step):
    df = random_groups(seed=11, n_groups=150, step=step)
    df.write_csv(tmp_path / "in.csv", null_value="NA")
    script = tmp_path / "run.R"
    script.write_text(R_SCRIPT)
    subprocess.run(
        ["Rscript", script, FUNCS_R, tmp_path / "in.csv", tmp_path / "out.csv"],
        check=True, capture_output=True,
    )
    r = pl.read_csv(tmp_path / "out.csv", null_values="NA")
    got = optimal_w(df.lazy(), "value", "bm", "g").collect()
    assert_series_equal(got["value_w"], r["value_w"].cast(pl.Float64))
