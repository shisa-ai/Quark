#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json

import pytest
import torch

from quark.shares.utils.import_utils import is_safetensors_available

if is_safetensors_available():
    from safetensors.torch import load_file, save_file


@pytest.mark.skipif(not is_safetensors_available(), reason="safetensors is required")
def test_real_quantized_quark_export_rewrites_internal_qparams_keys(tmp_path):
    from quark.torch.export.api import _rewrite_real_quantized_safetensor_keys_for_export

    shard_name = "model-00001-of-00001.safetensors"
    checkpoint_path = tmp_path / shard_name
    save_file(
        {
            "model.layers.0.mlp.down_proj.weight": torch.zeros((2, 2), dtype=torch.int8),
            "model.layers.0.mlp.down_proj.weight_quantizer.scale": torch.ones((2,), dtype=torch.float32),
            "model.layers.0.mlp.down_proj.weight_quantizer.zero_point": torch.zeros((2,), dtype=torch.int8),
            "model.layers.0.mlp.down_proj.input_quantizer.scale": torch.ones((1,), dtype=torch.float32),
        },
        str(checkpoint_path),
        metadata={"format": "pt"},
    )
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "model.layers.0.mlp.down_proj.weight": shard_name,
                    "model.layers.0.mlp.down_proj.weight_quantizer.scale": shard_name,
                    "model.layers.0.mlp.down_proj.weight_quantizer.zero_point": shard_name,
                    "model.layers.0.mlp.down_proj.input_quantizer.scale": shard_name,
                },
            }
        )
    )

    _rewrite_real_quantized_safetensor_keys_for_export(tmp_path)

    tensors = load_file(str(checkpoint_path))
    assert "model.layers.0.mlp.down_proj.weight" in tensors
    assert "model.layers.0.mlp.down_proj.weight_scale" in tensors
    assert "model.layers.0.mlp.down_proj.weight_zero_point" in tensors
    assert "model.layers.0.mlp.down_proj.input_scale" in tensors
    assert "model.layers.0.mlp.down_proj.weight_quantizer.scale" not in tensors

    index = json.loads(index_path.read_text())
    assert "model.layers.0.mlp.down_proj.weight_scale" in index["weight_map"]
    assert "model.layers.0.mlp.down_proj.weight_quantizer.scale" not in index["weight_map"]
