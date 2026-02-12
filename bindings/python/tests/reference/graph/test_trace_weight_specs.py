import torch
import torch.nn as nn

from . import utils as graph_utils


def _spec_by_id(specs: list[dict], name: str) -> dict:
    for spec in specs:
        if spec["id"] == name:
            return spec
    raise AssertionError(f"missing spec for {name}")


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("inv_freq", torch.ones(dim))


def test_extract_weight_specs_includes_layouts_and_overrides():
    import trace as trace_mod

    class SimpleBlock(nn.Module):
        weight_map_overrides = {
            "linear.weight": ["model.layers.{d}.custom.linear.weight"],
        }

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3, bias=False)
            self.embedding = nn.Embedding(10, 4)
            self.conv = nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=False)
            self.register_buffer("scale", torch.ones(4))
            self.rotary = RotaryEmbedding(4)

    _ = graph_utils  # Ensures tools/archs is on sys.path.

    specs = trace_mod.extract_weight_specs(SimpleBlock())

    linear_spec = _spec_by_id(specs, "linear.weight")
    assert linear_spec["module_type"] == "Linear"
    assert linear_spec["layout"] == "linear"
    assert linear_spec["candidates"][0] == "model.layers.{d}.custom.linear.weight"

    embedding_spec = _spec_by_id(specs, "embedding.weight")
    assert embedding_spec["module_type"] == "Embedding"
    assert embedding_spec["layout"] == "embedding"

    conv_spec = _spec_by_id(specs, "conv.weight")
    assert conv_spec["module_type"] == "Conv1d"
    assert conv_spec["layout"] == "conv1d_depthwise"

    buffer_spec = _spec_by_id(specs, "scale")
    assert buffer_spec["module_type"] == "SimpleBlock"
    assert buffer_spec["dtype"] == "float32"

    assert all(spec["id"] != "rotary.inv_freq" for spec in specs)
