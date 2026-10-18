from data_type import Value
from graphviz import Digraph
from PIL import Image
from io import BytesIO


def _value_id(var: Value) -> str:
    return f"v_{id(var)}"


def _op_id(var: Value) -> str:
    return f"op_{id(var)}"


def create_graph(var: Value) -> Digraph:
    dot = Digraph()
    visited: set[int] = set()

    def visit(v: Value) -> None:
        vid = id(v)
        if vid in visited:
            return
        visited.add(vid)

        if v.prev is None:
            name = v.label if v.label else "val"
            dot.node(
                _value_id(v),
                f"{v.data} | {v.grad}",
                shape="rectangle",
            )
            return

        for child in v.prev:
            visit(child)

        op = v.label or "?"
        dot.node(_op_id(v), op, shape="circle")
        dot.node(
            _value_id(v),
            f"{v.data} | {v.grad}",
            shape="rectangle",
        )

        for child in v.prev:
            dot.edge(_value_id(child), _op_id(v))
        dot.edge(_op_id(v), _value_id(v))

    visit(var)
    return dot


def show_graph(var: Value) -> None:
    dot = create_graph(var)
    img = Image.open(BytesIO(dot.pipe(format="png")))
    img.show()
