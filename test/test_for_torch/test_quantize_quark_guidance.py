#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import importlib.util
from pathlib import Path


_QUANTIZE_QUARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "torch"
    / "language_modeling"
    / "llm_ptq"
    / "quantize_quark.py"
)


def _load_quantize_quark_module():
    spec = importlib.util.spec_from_file_location("quantize_quark_example", _QUANTIZE_QUARK_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_int8_dynamic_guidance_says_calibration_dataset_is_unused_for_plain_scheme():
    module = _load_quantize_quark_module()
    args = argparse.Namespace(quant_scheme="int8_dynamic", quant_algo=None)

    guidance = module._get_quant_scheme_guidance(args)

    assert len(guidance) == 1
    assert "calibration dataset" in guidance[0]
    assert "do not determine quality-critical activation scales" in guidance[0]
    assert "skips loading" in guidance[0]
    assert "plain `int8_dynamic`" in guidance[0]


def test_int8_dynamic_guidance_keeps_algorithm_caveat():
    module = _load_quantize_quark_module()
    args = argparse.Namespace(quant_scheme="int8_dynamic", quant_algo=["smoothquant"])

    guidance = module._get_quant_scheme_guidance(args)

    assert len(guidance) == 1
    assert "dynamic runtime activation scales" in guidance[0]
    assert "may still use calibration samples" in guidance[0]


def test_static_int8_guidance_recommends_int8_dynamic():
    module = _load_quantize_quark_module()
    args = argparse.Namespace(quant_scheme="int8", quant_algo=None)

    guidance = module._get_quant_scheme_guidance(args)

    assert len(guidance) == 1
    assert "static W8A8 INT8" in guidance[0]
    assert "calibration data and sequence length" in guidance[0]
    assert "prefer `--quant_scheme int8_dynamic`" in guidance[0]


def test_other_scheme_has_no_guidance():
    module = _load_quantize_quark_module()
    args = argparse.Namespace(quant_scheme="fp8", quant_algo=None)

    assert module._get_quant_scheme_guidance(args) == []


def test_plain_int8_dynamic_skips_calibration_dataloader():
    module = _load_quantize_quark_module()
    args = argparse.Namespace(quant_scheme="int8_dynamic", quant_algo=None, model_export=["hf_format"])

    assert module._should_skip_calibration_dataloader(args)


def test_int8_dynamic_with_algorithm_keeps_calibration_dataloader():
    module = _load_quantize_quark_module()
    args = argparse.Namespace(quant_scheme="int8_dynamic", quant_algo=["smoothquant"], model_export=["hf_format"])

    assert not module._should_skip_calibration_dataloader(args)


def test_int8_dynamic_with_onnx_export_keeps_calibration_dataloader():
    module = _load_quantize_quark_module()
    args = argparse.Namespace(quant_scheme="int8_dynamic", quant_algo=None, model_export=["onnx"])

    assert not module._should_skip_calibration_dataloader(args)
