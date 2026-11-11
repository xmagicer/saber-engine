"""Pre-capture every accepted-row count and proposal shape before the socket opens."""

import time


WIDTHS = (4, 1, 3, 2)
CAPACITIES = (128, 512, 2048)


def exercise(model, features, anchors):
    """A single committed prefix crosses both context-capacity boundaries."""
    model.reset()
    position = 0
    for rows in (5, 1, 4, 2, 3):
        model.append(position, features[position : position + rows])
        position += rows
    picked = {}
    for cap, end, anchor in zip(CAPACITIES, (15, 129, 513), anchors):
        if end > position:
            model.append(position, features[position:end])
            position = end
        for width in WIDTHS:
            tokens = tuple(model.propose(position, anchor, width))
            if len(tokens) != width:
                raise RuntimeError(f"Expected {width} proposals at position {position}")
            picked[cap, width] = tokens
    return picked


def validate_prefixes(picked):
    for cap in CAPACITIES:
        for width in WIDTHS:
            if picked[cap, width] != picked[cap, 4][:width]:
                raise RuntimeError(
                    f"Proposal prefix differs at cap={cap}, width={width}"
                )


def prewarm_all(model):
    import torch

    if model.device.type != "cuda" or not model.graphs:
        raise ValueError("--prewarm-all requires GPU graph capture")

    def fixture(seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(513, 12800, generator=generator, dtype=torch.bfloat16)

    start = time.monotonic()
    try:
        exercise(model, fixture(171), (111, 527, 1522))
        if set(model.append_cache) != set(range(1, 6)):
            raise RuntimeError("Incomplete graph capture for accepted rows")
        if set(model.graph_cache) != {(cap, k) for cap in CAPACITIES for k in WIDTHS}:
            raise RuntimeError("Incomplete graph capture for proposal shapes")
        torch.cuda.synchronize(model.device)
        capture_s = time.monotonic() - start

        # Change both the committed states and the anchor after all graphs share a
        # pool. Replay the graphs out of width order, copy out each result, then
        # compare with a clean eager execution of precisely the same requests.
        holdout = fixture(329)
        anchors = (198, 1548, 9787)
        actual = exercise(model, holdout, anchors)
        model.graphs = False
        expected = exercise(model, holdout, anchors)
        for shape, result in actual.items():
            if result != expected[shape]:
                raise RuntimeError(
                    f"Graph/eager mismatch at {shape}: graph={result}, eager={expected[shape]}"
                )
        validate_prefixes(expected)
        return {
            "capture_s": capture_s,
            "validation_s": time.monotonic() - start - capture_s,
            "append_rows": sorted(model.append_cache),
            "proposal_shapes": sorted(model.graph_cache),
            "eager_ids_equal": True,
            "width_prefixes_equal": True,
        }
    finally:
        model.graphs = True
        model.reset()
        torch.cuda.synchronize(model.device)
