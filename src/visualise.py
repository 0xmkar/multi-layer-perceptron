from data_type import Value
from graphviz import Digraph
from mlp import multi_layer_perceptron
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

# AI slop to visualise the MLP
def _fmt(var: Value) -> str:
    return f"{var.data:.2f} | {var.grad:.2f}"


def _weight_edge(weight: Value) -> dict[str, str]:
    magnitude = min(abs(weight.data), 1.0)
    return {
        "color": "#1a5276" if weight.data >= 0 else "#922b21",
        "penwidth": f"{0.5 + 2.2 * magnitude:.2f}",
        "arrowsize": "0.55",
    }


def _ancestor_ids(var: Value) -> set[int]:
    seen: set[int] = set()

    def walk(v: Value) -> None:
        vid = id(v)
        if vid in seen:
            return
        seen.add(vid)
        if v.prev:
            for child in v.prev:
                walk(child)

    walk(var)
    return seen


def _values_in_graph(roots: list[Value]) -> list[Value]:
    seen: set[int] = set()
    ordered: list[Value] = []

    def walk(v: Value) -> None:
        vid = id(v)
        if vid in seen:
            return
        seen.add(vid)
        if v.prev:
            for child in v.prev:
                walk(child)
        ordered.append(v)

    for root in roots:
        walk(root)
    return ordered


def activations_from_forward(
    mlp: multi_layer_perceptron,
    inputs: list[Value],
    outputs: list[Value],
) -> list[list[Value]]:
    """Recover per-layer activations from the graph of a prior ``mlp(inputs)`` call."""
    if len(outputs) != mlp.layers[-1].n_out:
        raise ValueError(
            f"expected {mlp.layers[-1].n_out} outputs, got {len(outputs)}"
        )
    if len(inputs) != mlp.layers[0].n_in:
        raise ValueError(f"expected {mlp.layers[0].n_in} inputs, got {len(inputs)}")

    reachable = _values_in_graph(outputs)
    activations: list[list[Value]] = [inputs]
    used_tanh: set[int] = set()

    for layer in mlp.layers:
        layer_out: list[Value] = []
        for neuron in layer.neurons:
            bias_id = id(neuron.b)
            match: Value | None = None
            for var in reachable:
                if var.op != "tanh" or id(var) in used_tanh:
                    continue
                if bias_id in _ancestor_ids(var):
                    match = var
                    break
            if match is None:
                raise ValueError(
                    "no activation graph on this MLP; call mlp(inputs) before show_mlp"
                )
            used_tanh.add(id(match))
            layer_out.append(match)
        activations.append(layer_out)

    return activations


def _patch_mlp_forward_cache() -> None:
    if getattr(multi_layer_perceptron.__call__, "_visualise_cache", False):
        return

    original = multi_layer_perceptron.__call__

    def __call__(self: multi_layer_perceptron, x: list[Value]) -> list[Value]:
        out = original(self, x)
        self._last_inputs = x
        self._last_outputs = out
        return out

    __call__._visualise_cache = True  # type: ignore[attr-defined]
    multi_layer_perceptron.__call__ = __call__  # type: ignore[method-assign]


_patch_mlp_forward_cache()


def create_mlp_graph(
    mlp: multi_layer_perceptron,
    activations: list[list[Value]] | None = None,
) -> Digraph:
    # neato with pinned coordinates. dot reorders nodes to shorten edges,
    # which scrambled neuron indices inside each layer.
    y_gap = 1.7
    x_gap = 3.6
    columns: list[tuple[str, str, str, str, list[tuple[str, str]]]] = []

    n_in = mlp.layers[0].n_in
    if activations is not None:
        if len(activations[0]) != n_in:
            raise ValueError(
                f"expected {n_in} inputs, got {len(activations[0])}"
            )
        input_nodes = [
            (f"in_{i}", f"x{i}\\n{_fmt(activations[0][i])}") for i in range(n_in)
        ]
    else:
        input_nodes = [(f"in_{i}", f"x{i}") for i in range(n_in)]
    columns.append((
        "cluster_input",
        "Input",
        "#eaf3fb",
        "#2471a3",
        input_nodes,
    ))

    hidden_i = 0
    last = len(mlp.layers) - 1
    for li, layer in enumerate(mlp.layers):
        if li == last:
            title, fill, border, name = "Output", "#e8f8f0", "#1e8449", "cluster_output"
        else:
            hidden_i += 1
            title = f"Hidden {hidden_i}"
            fill, border, name = "#fef5e7", "#b9770e", f"cluster_hidden_{hidden_i}"
        if activations is not None:
            layer_out = activations[li + 1]
            neuron_labels = [
                (f"L{li}N{ni}", f"{ni}\\n{_fmt(layer_out[ni])}")
                for ni in range(len(layer.neurons))
            ]
        else:
            neuron_labels = [
                (f"L{li}N{ni}", str(ni)) for ni in range(len(layer.neurons))
            ]
        columns.append((name, title, fill, border, neuron_labels))

    max_n = max(len(nodes) for *_, nodes in columns)
    dot = Digraph(
        engine="neato",
        graph_attr={
            "splines": "true",
            "overlap": "scale",
            "pad": "0.3",
            "fontname": "Helvetica",
        },
        node_attr={"fontname": "Helvetica", "fontsize": "11"},
        edge_attr={"arrowsize": "0.55"},
    )

    prev_ids: list[str] | None = None
    for li, (name, title, fill, border, nodes) in enumerate(columns):
        n = len(nodes)
        x = li * x_gap
        with dot.subgraph(name=name) as cluster:
            cluster.attr(
                label=title,
                labelloc="t",
                style="rounded,filled",
                fillcolor=fill,
                color=border,
                penwidth="2",
                fontsize="15",
                fontname="Helvetica",
                margin="0.3",
            )
            y0 = (max_n - 1) * y_gap / 2 - (max_n - n) * y_gap / 2
            # Empty point above this column so the layer title clears neuron 0.
            cluster.node(
                f"{name}_pad",
                "",
                shape="point",
                style="invis",
                width="0.01",
                height="0.01",
                pos=f"{x},{y0 + 1.05}!",
                pin="true",
            )
            for i, (node_id, label) in enumerate(nodes):
                cluster.node(
                    node_id,
                    label,
                    shape="circle",
                    style="filled",
                    fillcolor="white",
                    color=border,
                    penwidth="1.8",
                    width="1.15",
                    height="1.15",
                    fixedsize="true",
                    pos=f"{x},{y0 - i * y_gap}!",
                    pin="true",
                )
        if prev_ids is not None:
            layer = mlp.layers[li - 1]
            for ni, neuron in enumerate(layer.neurons):
                for wi, weight in enumerate(neuron.w):
                    dot.edge(prev_ids[wi], f"L{li - 1}N{ni}", **_weight_edge(weight))
        prev_ids = [node_id for node_id, _ in nodes]

    return dot


def show_mlp(
    mlp: multi_layer_perceptron,
    x: list[Value] | None = None,
    *,
    outputs: list[Value] | None = None,
) -> None:
    """Visualise MLP structure using the graph from the last ``mlp(inputs)`` call.

    Does not run another forward pass. After ``b = a(inputs)`` and ``loss.backwards()``,
    node labels show each activation's ``data | grad`` from those same ``Value`` objects.
    Optional ``x`` / ``outputs`` override the cached last forward on ``mlp``.
    """
    inputs = x if x is not None else getattr(mlp, "_last_inputs", None)
    roots = outputs if outputs is not None else getattr(mlp, "_last_outputs", None)

    activations: list[list[Value]] | None = None
    if inputs is not None and roots is not None:
        activations = activations_from_forward(mlp, inputs, roots)

    dot = create_mlp_graph(mlp, activations)
    img = Image.open(BytesIO(dot.pipe(format="png")))
    img.show()
