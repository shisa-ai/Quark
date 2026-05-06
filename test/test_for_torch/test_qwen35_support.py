#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy

import pytest
import torch

from quark.torch import LLMTemplate
from quark.torch.quantization.file2file_quantization import _get_layer_quant_config_by_tensor_name
from quark.torch.utils.llm.model_preparation import prepare_for_moe_quant
from quark.torch.utils.llm.module_replacement.replacement_utils import replace_qwen3_5_moe_experts_with_linear


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"])
def test_qwen35_templates_exclude_mtp_weights_for_file2file(model_type: str):
    """Qwen3.5/3.6 checkpoints contain auxiliary MTP weights that vLLM/HF loaders skip.

    File-to-file quantization should therefore copy those tensors through unchanged instead of partially
    quantizing them.
    """
    quant_config = LLMTemplate.get(model_type).get_config("int4_wo_128")
    excluded_layer_names: set[str] = set()

    for tensor_name in [
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
    ]:
        assert (
            _get_layer_quant_config_by_tensor_name(
                tensor_name=tensor_name,
                quant_config=quant_config,
                excluded_layer_names=excluded_layer_names,
            )
            is None
        )

    assert excluded_layer_names == {
        "mtp.fc",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.mlp.down_proj",
        "mtp.layers.0.mlp.gate_proj",
        "mtp.layers.0.mlp.up_proj",
    }

    # Non-MTP language weights should still be eligible for quantization.
    assert (
        _get_layer_quant_config_by_tensor_name(
            tensor_name="model.language_model.layers.0.mlp.down_proj.weight",
            quant_config=quant_config,
        )
        is not None
    )


def _get_qwen35_moe_classes():
    transformers = pytest.importorskip("transformers")
    try:
        from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"Transformers build does not provide Qwen3.5-MoE classes: {exc}")

    return transformers, Qwen3_5MoeTextConfig, Qwen3_5MoeExperts, Qwen3_5MoeForCausalLM


def test_qwen35_moe_expert_replacement_matches_fused_forward_cpu():
    """The replacement used for quantization should preserve fused expert numerics on CPU."""
    _, Qwen3_5MoeTextConfig, Qwen3_5MoeExperts, _ = _get_qwen35_moe_classes()

    torch.manual_seed(0)
    config = Qwen3_5MoeTextConfig(
        hidden_size=8,
        moe_intermediate_size=4,
        num_experts=3,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    experts = Qwen3_5MoeExperts(config).eval()
    with torch.no_grad():
        experts.gate_up_proj.normal_()
        experts.down_proj.normal_()

    original_experts = copy.deepcopy(experts).eval()
    hidden_states = torch.randn(5, 8)
    selected_experts = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]])
    routing_weights = torch.rand(5, 2)

    expected = original_experts(hidden_states, selected_experts, routing_weights)
    replace_qwen3_5_moe_experts_with_linear(experts)
    actual = experts(hidden_states, selected_experts, routing_weights)

    assert not hasattr(experts, "gate_up_proj")
    assert not hasattr(experts, "down_proj")
    assert hasattr(getattr(experts, "0"), "gate_proj")
    assert hasattr(getattr(experts, "0"), "up_proj")
    assert hasattr(getattr(experts, "0"), "down_proj")
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_prepare_for_moe_quant_replaces_qwen35_moe_text_experts_cpu():
    """Exercise the model_preparation hook for text-only Qwen3.5-MoE configs without a GPU."""
    _, Qwen3_5MoeTextConfig, _, Qwen3_5MoeForCausalLM = _get_qwen35_moe_classes()

    config = Qwen3_5MoeTextConfig(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=3,
        num_experts_per_tok=2,
        vocab_size=100,
        layer_types=["full_attention"],
    )
    model = Qwen3_5MoeForCausalLM(config).eval()
    experts = model.model.layers[0].mlp.experts
    assert hasattr(experts, "gate_up_proj")

    prepare_for_moe_quant(model)

    experts = model.model.layers[0].mlp.experts
    assert not hasattr(experts, "gate_up_proj")
    assert not hasattr(experts, "down_proj")
    assert hasattr(getattr(experts, "0"), "gate_proj")
