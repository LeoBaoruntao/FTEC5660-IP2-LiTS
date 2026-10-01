"""Check bfs.py::BFSSearch terminal-count conditions without model inference.

Evaluate the actual source expressions at boundary counts; this is a focused
condition regression, not an end-to-end search test.
"""

import ast
from pathlib import Path

from lits.agents.tree.bfs import BFSConfig


def main():
    """Verify all four stopping checks below, at, and above configured thresholds."""
    path = Path(__file__).resolve().parents[2] / "lits/agents/tree/bfs.py"
    tree = ast.parse(path.read_text())
    checks = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
              and any(isinstance(value, ast.Attribute) and value.attr == "max_leaves_to_terminate"
                      for value in node.comparators)]
    if len(checks) != 4:
        raise RuntimeError("Review changed BFS stopping-check coverage")
    for threshold in (0, 1, 5, 11):
        config = BFSConfig(max_leaves_to_terminate=threshold)
        for check in checks:
            expression = compile(ast.Expression(check), str(path), "eval")
            for count in {max(0, threshold - 1), threshold, threshold + 1}:
                actual = eval(expression, {"terminal_nodes": [None] * count, "config": config})
                if actual != (count >= threshold):
                    raise RuntimeError(f"Wrong threshold boundary at source line {check.lineno}")
    print("Passed all four BFS terminal-count conditions at threshold boundaries")
    breakpoint()  # inspect: checks, config


if __name__ == "__main__":
    main()
