"""Collection of the core mathematical operators used throughout the code base."""

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# TODO: Implement for Task 0.1.
import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y


def id(x: float) -> float:
    """Returns the identity of a number."""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negates a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Compares two numbers."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid of a number."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculates the ReLU of a number."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural logarithm of a number."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential of a number."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the inverse of a number."""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Calculates the derivative of the logarithm of a number."""
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Calculates the derivative of the inverse of a number."""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Calculates the derivative of the ReLU of a number."""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        f: Function from one value to one value.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element, and returns a new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(f(x))
        return ret

    return _map


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        f: combine two values

    Returns:
    -------
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by applying fn(x, y) on each pair of elements.

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(f(x, y))
        return ret

    return _zipWith


def reduce(
    f: Callable[[float, float], float], initial: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
    ----
        f: combine two values
        initial: start value $x_0$

    Returns:
    -------
        Function that takes a list `ls` of elements
        $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2, fn(x_1, x_0)))`

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = initial
        for x in ls:
            val = f(val, x)
        return val

    return _reduce


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negates a list."""
    return map(neg)(xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Adds two lists together."""
    return zipWith(add)(xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Calculates the sum of a list."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Calculates the product of a list."""
    return reduce(mul, 1.0)(xs)
