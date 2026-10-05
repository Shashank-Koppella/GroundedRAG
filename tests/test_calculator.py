"""Tests for src/agent/calculator.py."""
import pytest

from src.agent.calculator import CalcError, calculate


def v(expr):
    return calculate(expr).value


# ---------------------------------------------------------------- arithmetic

def test_basic_arithmetic_and_precedence():
    assert v("1 + 2 * 3") == 7
    assert v("(1 + 2) * 3") == 9
    assert v("-4 + 10") == 6
    assert v("2 ** 3 ** 2") == 512  # right-associative, exponent 9 is within the cap
    assert v("10 / 4") == 2.5


def test_arithmetic_is_exact_not_floating_point():
    r = calculate("0.1 + 0.2")
    assert r.exact == "3/10" and r.value == 0.3
    assert calculate("1 / 3 * 3").exact == "1"


def test_large_dollar_amounts_stay_exact():
    r = calculate("416161000000 - 391035000000")
    assert r.exact == "25126000000" and r.display == "25,126,000,000"


# ---------------------------------------------------------------- finance helpers

def test_pct_change_known_values():
    assert v("pct_change(100, 110)") == pytest.approx(10.0)
    assert v("pct_change(200, 150)") == pytest.approx(-25.0)
    assert v("pct_change(391035000000, 416161000000)") == pytest.approx((416161 - 391035) / 391035 * 100)


def test_pct_change_refuses_a_zero_or_negative_base():
    with pytest.raises(CalcError, match="zero"):
        calculate("pct_change(0, 5)")
    with pytest.raises(CalcError, match="negative"):
        calculate("pct_change(-2722000000, 30425000000)")  # AMZN FY2022 net income is negative


def test_pct_margin_ratio_mean():
    assert v("pct(25, 200)") == 12.5
    assert v("ratio(10, 4)") == 2.5
    assert v("mean(1, 2, 3, 4)") == 2.5
    assert calculate("mean(1, 2)").exact == "3/2"


def test_cagr():
    assert v("cagr(100, 121, 2)") == pytest.approx(10.0)
    with pytest.raises(CalcError):
        calculate("cagr(-5, 10, 2)")
    with pytest.raises(CalcError):
        calculate("cagr(5, 10, 0)")


def test_round_abs_min_max():
    assert v("round(2.345, 2)") == 2.34  # exact decimal tie (2345/1000) rounds to even, no float surprise
    assert v("round(7.5)") == 8 and v("round(6.5)") == 6  # ties to even, exactly
    assert v("abs(-3)") == 3 and v("min(4, 2, 9)") == 2 and v("max(4, 2, 9)") == 9
    with pytest.raises(CalcError):
        calculate("round(1.5, 0.5)")


def test_nested_calls():
    assert v("round(pct_change(100, 133), 1)") == 33.0


# ---------------------------------------------------------------- refusals

@pytest.mark.parametrize("expr", ["1 / 0", "0 ** -1", "ratio(5, 0)", "pct(5, 0)"])
def test_division_by_zero_is_a_calc_error(expr):
    with pytest.raises(CalcError, match="division by zero"):
        calculate(expr)


@pytest.mark.parametrize("expr", ["2 ** 13", "2 ** -13", "2 ** 0.5", "10 ** 10 ** 10"])
def test_exponent_limits(expr):
    with pytest.raises(CalcError):
        calculate(expr)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('x')",
    "().__class__",
    "[1, 2][0]",
    "lambda: 1",
    "'abc'",
    "True + 1",
    "x + 1",
    "abs.__name__",
    "1 if 2 else 3",
    "1 < 2",
    "pct_change(old=1, new=2)",
    "unknown(1)",
    "1e400",
    "$100 + 5",
    "5%",
    "1,000 + 1",
])
def test_everything_outside_the_whitelist_is_refused(expr):
    with pytest.raises(CalcError):
        calculate(expr)


def test_empty_overlong_and_oversized_expressions_are_refused():
    for bad in ["", "   ", None, 5]:
        with pytest.raises(CalcError):
            calculate(bad)
    with pytest.raises(CalcError, match="too long"):
        calculate("1 + " * 150 + "1")
    with pytest.raises(CalcError, match="too complex"):
        calculate("+".join(["1"] * 100))


def test_wrong_argument_count_is_a_calc_error_not_a_crash():
    with pytest.raises(CalcError):
        calculate("pct_change(1)")
    with pytest.raises(CalcError):
        calculate("mean()")


def test_result_dict_shape():
    d = calculate(" 6 / 4 ").as_dict()
    assert d == {"expression": "6 / 4", "value": 1.5, "exact": "3/2", "display": "1.5"}

# ---------------------------------------------------------------- audit fixes (Oct 5)

@pytest.mark.parametrize("expr", ["((10**12)**12)**12", "1e308*10", "9" * 350,
                                  "((((((((2**12)**12)**12)**12)**12)**12)**12)**12)"])
def test_huge_values_are_a_calc_error_not_an_overflow_or_a_memory_blowup(expr):
    import time
    t = time.time()
    with pytest.raises(CalcError, match="too large"):
        calculate(expr)
    assert time.time() - t < 0.5     # bounded BEFORE the power is materialised


@pytest.mark.parametrize("expr", ["min()", "max()"])
def test_empty_min_max_are_calc_errors(expr):
    with pytest.raises(CalcError, match="at least one value"):
        calculate(expr)


def test_realistic_magnitudes_are_unaffected_by_the_size_cap():
    assert calculate("416161000000 * 416161000000").exact == str(416161000000 ** 2)
    assert v("pct_change(391035000000, 416161000000)") == pytest.approx(6.4255, abs=1e-4)
