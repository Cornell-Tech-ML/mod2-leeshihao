from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Function for accumulating the derivative of `self` with respect to `x`."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`), False otherwise."""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant (created by the user)."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """The immediate parents of this variable, in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the derivatives of this node with respect to each leaf variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    sorted_vars: List[Variable] = []

    def dfs(var: Variable) -> None:
        """DFS for topological sorting"""
        if var.unique_id in visited or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    dfs(parent)
        visited.add(var.unique_id)
        sorted_vars.insert(0, var)

    dfs(variable)
    return sorted_vars


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Send `deriv` back through the graph to accumulate derivatives."""
    # TODO: Implement for Task 1.4.
    sorted_vars = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for var in sorted_vars:
        d = derivatives.get(var.unique_id, 0)
        if var.is_leaf():
            var.accumulate_derivative(d)
        else:
            for parent, grad in var.chain_rule(d):
                if parent.is_constant():
                    continue
                derivatives.setdefault(parent.unique_id, 0)
                derivatives[parent.unique_id] = derivatives[parent.unique_id] + grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns a tuple containing any tensors that `save_for_backward` saves for use in backprop."""
        return self.saved_values
