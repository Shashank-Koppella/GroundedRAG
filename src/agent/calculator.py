"""
Calculator tool -- Phase B, Day 8.

Why this exists: Day 7 found NLI cannot verify arithmetic, and the generator prompt forbids the
model from computing new numbers ("Never round, convert, or calculate"). Every derived figure
(growth rate, margin, ratio, average) must therefore come from here, not from the LLM.

Design choices
  * Never `eval`. Expressions are parsed with `ast` and only a whitelist of nodes is walked:
    numbers, + - * / **, unary +/-, parentheses, and a fixed set of named functions.
  * Exact arithmetic: all values are `fractions.Fraction`, so 0.1 + 0.2 == 0.3 and a division like
    revenue ratios never picks up float noise. The final value is converted to float once, for
    display. (`cagr` needs a fractional power and is the one place a float is used.)
  * Refuses instead of guessing: division by zero, a percentage change off a non-positive base
    (e.g. AMZN's FY2022 net income is -2.7B: "percent change" from there is meaningless), a
    non-integer or oversized exponent, unknown names, and over-long expressions all raise CalcError
    with a message the agent can pass back to the user.
  * Inputs must be plain numbers in base units (whole dollars). No "$", "%", commas or words;
    guessing at units is exactly the kind of silent error this tool is here to remove.
"""
import ast
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict

MAX_EXPR_CHARS = 400
MAX_ABS_EXPONENT = 12
MAX_NODES = 120
MAX_BITS = 1000          # cap on numerator/denominator size of every intermediate value (~1e301):
                         # keeps the float conversion finite and stops (2**12)**12... from eating memory


class CalcError(ValueError):
    """The expression is invalid or the operation is not meaningful. Message is user-safe."""


def _bounded(v: Fraction) -> Fraction:
    if v.numerator.bit_length() > MAX_BITS or v.denominator.bit_length() > MAX_BITS:
        raise CalcError("a value in the expression is too large")
    return v


def _num(x) -> Fraction:
    if isinstance(x, Fraction):
        return _bounded(x)
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise CalcError(f"not a number: {x!r}")
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        raise CalcError("non-finite number")
    return _bounded(Fraction(str(x)) if isinstance(x, float) else Fraction(x))


def pct_change(old, new) -> Fraction:
    """(new - old) / old * 100. Refuses a non-positive base."""
    old, new = _num(old), _num(new)
    if old == 0:
        raise CalcError("percentage change is undefined when the starting value is zero")
    if old < 0:
        raise CalcError("percentage change is not meaningful when the starting value is negative; "
                        "report the two values and the absolute change instead")
    return (new - old) / old * 100


def pct(part, whole) -> Fraction:
    """part / whole * 100 (a margin or share)."""
    part, whole = _num(part), _num(whole)
    if whole == 0:
        raise CalcError("division by zero")
    return part / whole * 100


def ratio(a, b) -> Fraction:
    a, b = _num(a), _num(b)
    if b == 0:
        raise CalcError("division by zero")
    return a / b


def mean(*xs) -> Fraction:
    if not xs:
        raise CalcError("mean() needs at least one value")
    vals = [_num(x) for x in xs]
    return sum(vals, Fraction(0)) / len(vals)


def cagr(begin, end, years) -> Fraction:
    """Compound annual growth rate, in percent. Uses a float power (the only inexact step)."""
    begin, end, years = _num(begin), _num(end), _num(years)
    if begin <= 0 or end <= 0:
        raise CalcError("cagr needs positive begin and end values")
    if years <= 0:
        raise CalcError("cagr needs a positive number of years")
    return _num(round(((float(end) / float(begin)) ** (1 / float(years)) - 1) * 100, 10))


def _abs(x): return abs(_num(x))
def _min(*xs):
    if not xs:
        raise CalcError("min() needs at least one value")
    return min(_num(x) for x in xs)


def _max(*xs):
    if not xs:
        raise CalcError("max() needs at least one value")
    return max(_num(x) for x in xs)


def _round(x, n=0):
    n = _num(n)
    if n.denominator != 1 or not (0 <= n <= 12):
        raise CalcError("round() digits must be an integer between 0 and 12")
    return _num(round(_num(x), int(n)))  # exact Fraction rounding, ties to even


FUNCTIONS: Dict[str, Callable] = {
    "pct_change": pct_change, "pct": pct, "ratio": ratio, "mean": mean, "cagr": cagr,
    "abs": _abs, "min": _min, "max": _max, "round": _round,
}

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
}


def _eval(node, budget):
    budget[0] += 1
    if budget[0] > MAX_NODES:
        raise CalcError("expression is too complex")
    if isinstance(node, ast.Expression):
        return _eval(node.body, budget)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError("only plain numbers are allowed")
        return _num(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand, budget)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        if type(node.op) in _BINOPS:
            return _bounded(_BINOPS[type(node.op)](_eval(node.left, budget), _eval(node.right, budget)))
        if isinstance(node.op, ast.Div):
            l, r = _eval(node.left, budget), _eval(node.right, budget)
            if r == 0:
                raise CalcError("division by zero")
            return _bounded(l / r)
        if isinstance(node.op, ast.Pow):
            base, exp = _eval(node.left, budget), _eval(node.right, budget)
            if exp.denominator != 1 or abs(exp) > MAX_ABS_EXPONENT:
                raise CalcError(f"exponent must be an integer between -{MAX_ABS_EXPONENT} and "
                                f"{MAX_ABS_EXPONENT} (use cagr() for fractional growth powers)")
            if base == 0 and exp < 0:
                raise CalcError("division by zero")
            # bound the size BEFORE computing the power, so a huge result is never materialised
            if max(base.numerator.bit_length(), base.denominator.bit_length()) * abs(int(exp)) > MAX_BITS:
                raise CalcError("a value in the expression is too large")
            return _bounded(base ** int(exp))
        raise CalcError("operator not allowed")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            raise CalcError(f"unknown function; available: {', '.join(sorted(FUNCTIONS))}")
        if node.keywords:
            raise CalcError("keyword arguments are not allowed")
        args = [_eval(a, budget) for a in node.args]
        try:
            return _num(FUNCTIONS[node.func.id](*args))
        except TypeError as e:
            raise CalcError(f"wrong arguments for {node.func.id}()") from e
    raise CalcError("unsupported syntax; use numbers, + - * / **, parentheses and the named functions")


@dataclass(frozen=True)
class CalcResult:
    expression: str
    value: float            # for display / comparison
    exact: str              # exact rational, e.g. "1/3"
    display: str

    def as_dict(self):
        return {"expression": self.expression, "value": self.value, "exact": self.exact, "display": self.display}


def _display(v: Fraction) -> str:
    f = float(v)
    if v.denominator == 1:
        return f"{int(v):,}"
    return f"{f:,.4f}".rstrip("0").rstrip(".")


def calculate(expression: str) -> CalcResult:
    if not isinstance(expression, str) or not expression.strip():
        raise CalcError("empty expression")
    if len(expression) > MAX_EXPR_CHARS:
        raise CalcError("expression is too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as e:
        raise CalcError("could not parse the expression") from e
    v = _eval(tree, [0])
    return CalcResult(expression=expression.strip(), value=float(v), exact=str(v), display=_display(v))
