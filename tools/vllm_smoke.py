#!/usr/bin/env python3
"""vLLM smoke-test harness for HF and Quark-exported LLMs.

Examples:

  # Environment/import sanity only.
  python tools/vllm_smoke.py --check-env

  # Baseline unquantized Qwen smoke tests.
  python tools/vllm_smoke.py --suite qwen-baseline --enforce-eager

  # One model by HF id.
  python tools/vllm_smoke.py --model Qwen/Qwen3.5-0.8B

  # One local Quark export. Quantization is usually auto-detected from config.json;
  # pass --quantization quark if vLLM does not infer it.
  python tools/vllm_smoke.py --model qwen35-int8=/path/to/exported_model --quantization quark

Notes:
  - Run from a vLLM environment, not the Quark development env.
  - On ROCm, torch.cuda.* is still the PyTorch namespace; check torch.version.hip.
  - Qwen3.5/Qwen3.6 dense/MoE use hybrid cache paths, so this script auto-passes
    mamba_cache_mode='align' when it can identify qwen3_5 configs.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PRESETS: dict[str, dict[str, Any]] = {
    "qwen3-0.6b": {
        "label": "qwen3-0.6b",
        "model": "Qwen/Qwen3-0.6B",
        "dtype": "bfloat16",
        "mamba_cache_mode": None,
    },
    "qwen35-0.8b": {
        "label": "qwen35-0.8b",
        "model": "Qwen/Qwen3.5-0.8B",
        "dtype": "bfloat16",
        "mamba_cache_mode": "align",
    },
}

SUITES: dict[str, list[str]] = {
    "qwen-baseline": ["qwen3-0.6b", "qwen35-0.8b"],
}


@dataclass
class SmokeResult:
    label: str
    model: str
    ok: bool
    load_seconds: float | None = None
    generate_seconds: float | None = None
    output: str | None = None
    error: str | None = None


def _parse_value(raw: str) -> Any:
    """Parse a CLI KEY=VALUE value into bool/int/float/JSON/string."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _parse_key_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"Expected non-empty KEY in {raw!r}")
    return key, _parse_value(value)


def _parse_model_arg(raw: str) -> dict[str, Any]:
    """Parse PRESET or [LABEL=]MODEL_OR_PATH."""
    if raw in PRESETS:
        return dict(PRESETS[raw])

    if "=" in raw:
        label, model = raw.split("=", 1)
        if not label or not model:
            raise argparse.ArgumentTypeError(f"Expected LABEL=MODEL, got {raw!r}")
        return {"label": label, "model": model}

    model = raw
    label = Path(raw).name if Path(raw).exists() else raw.replace("/", "_")
    return {"label": label, "model": model}


def _read_model_type(model: str) -> str | None:
    """Read model_type for local model directories when possible."""
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        model_type = config.get("model_type")
        return str(model_type) if model_type else None
    except Exception:
        return None


def _is_qwen35_like(model: str) -> bool:
    model_lower = model.lower()
    if "qwen3.5" in model_lower or "qwen3.6" in model_lower or "qwen35" in model_lower:
        return True
    model_type = _read_model_type(model)
    return model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}


def _print_presets() -> None:
    print("Presets:")
    for name, spec in sorted(PRESETS.items()):
        print(f"  {name:14s} -> {spec['model']}")
    print("\nSuites:")
    for name, presets in sorted(SUITES.items()):
        print(f"  {name:14s} -> {', '.join(presets)}")


def _check_env() -> None:
    import torch
    import vllm

    print(f"python: {sys.executable}")
    print(f"vllm: {getattr(vllm, '__version__', '<unknown>')}")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.version.hip: {torch.version.hip}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    if torch.cuda.device_count():
        print(f"torch.cuda.device_name[0]: {torch.cuda.get_device_name(0)}")
    import vllm._C  # noqa: F401

    try:
        import vllm._rocm_C  # noqa: F401

        print("vllm._rocm_C: OK")
    except Exception as exc:
        print(f"vllm._rocm_C: unavailable ({exc!r})")
    print("vllm._C: OK")


def _build_llm_kwargs(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": spec["model"],
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }

    dtype = args.dtype or spec.get("dtype")
    if dtype:
        kwargs["dtype"] = dtype

    if args.quantization:
        kwargs["quantization"] = args.quantization

    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len

    if args.enforce_eager:
        kwargs["enforce_eager"] = True

    if args.cpu_offload_gb:
        kwargs["cpu_offload_gb"] = args.cpu_offload_gb

    mamba_cache_mode = args.mamba_cache_mode
    if mamba_cache_mode == "auto":
        mamba_cache_mode = spec.get("mamba_cache_mode")
        if mamba_cache_mode is None and _is_qwen35_like(spec["model"]):
            mamba_cache_mode = "align"
    if mamba_cache_mode and mamba_cache_mode != "none":
        kwargs["mamba_cache_mode"] = mamba_cache_mode

    for key, value in args.engine_arg:
        kwargs[key] = value

    return kwargs


def _run_one(spec: dict[str, Any], args: argparse.Namespace) -> SmokeResult:
    label = spec["label"]
    model = spec["model"]
    print(f"\n=== [{label}] {model} ===", flush=True)

    try:
        import torch
        from vllm import LLM, SamplingParams

        kwargs = _build_llm_kwargs(spec, args)
        if args.print_kwargs:
            printable = dict(kwargs)
            printable["model"] = model
            print("LLM kwargs:", json.dumps(printable, indent=2, sort_keys=True, default=str), flush=True)

        t0 = time.perf_counter()
        llm = LLM(**kwargs)
        load_seconds = time.perf_counter() - t0
        print(f"LOAD OK in {load_seconds:.2f}s", flush=True)

        generated_text: str | None = None
        generate_seconds: float | None = None
        if not args.load_only:
            sampling_kwargs: dict[str, Any] = {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            if args.top_p is not None:
                sampling_kwargs["top_p"] = args.top_p
            sampling_params = SamplingParams(**sampling_kwargs)

            t1 = time.perf_counter()
            outputs = llm.generate([args.prompt], sampling_params)
            generate_seconds = time.perf_counter() - t1
            generated_text = outputs[0].outputs[0].text
            print(f"GENERATE OK in {generate_seconds:.2f}s", flush=True)
            print(f"PROMPT: {args.prompt!r}", flush=True)
            print(f"OUTPUT: {generated_text!r}", flush=True)

        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return SmokeResult(
            label=label,
            model=model,
            ok=True,
            load_seconds=load_seconds,
            generate_seconds=generate_seconds,
            output=generated_text,
        )
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
        if args.traceback:
            traceback.print_exc()
        return SmokeResult(label=label, model=model, ok=False, error=error)


def _resolve_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for suite in args.suite:
        if suite not in SUITES:
            raise SystemExit(f"Unknown suite {suite!r}; known suites: {', '.join(sorted(SUITES))}")
        specs.extend(dict(PRESETS[preset]) for preset in SUITES[suite])
    specs.extend(_parse_model_arg(model_arg) for model_arg in args.model)
    return specs


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test vLLM model loading/generation for HF ids or local Quark exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true", help="List presets/suites and exit.")
    parser.add_argument("--check-env", action="store_true", help="Print torch/vLLM ROCm environment info and exit.")
    parser.add_argument("--suite", action="append", default=[], help=f"Preset suite. Known: {', '.join(sorted(SUITES))}")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model preset, HF id, local path, or LABEL=HF_ID_OR_PATH. Can be repeated.",
    )

    parser.add_argument("--prompt", default="Write one sentence about ROCm.")
    parser.add_argument("--load-only", action="store_true", help="Only instantiate LLM; skip generation.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)

    parser.add_argument("--dtype", default=None, help="Override dtype, e.g. auto, bfloat16, float16.")
    parser.add_argument("--quantization", default=None, help="Optional vLLM quantization override, e.g. quark.")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--mamba-cache-mode",
        choices=["auto", "align", "none"],
        default="auto",
        help="Use align for Qwen3.5/Qwen3.6 by default; use none to omit.",
    )
    parser.add_argument(
        "--engine-arg",
        action="append",
        type=_parse_key_value,
        default=[],
        metavar="KEY=VALUE",
        help="Extra vLLM LLM(...) kwarg. Can be repeated.",
    )

    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--print-kwargs", action="store_true")
    parser.add_argument("--traceback", action="store_true", help="Print full traceback on failure.")
    parser.add_argument("--output-json", default=None, help="Optional path to write JSON results.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.list:
        _print_presets()
        return 0

    if args.check_env:
        _check_env()
        return 0

    specs = _resolve_specs(args)
    if not specs:
        print("No models selected. Use --list, --suite qwen-baseline, or --model MODEL.", file=sys.stderr)
        return 2

    # Quick API visibility for debugging across vLLM releases.
    if args.print_kwargs:
        try:
            from vllm import LLM

            print("LLM.__init__ signature:", inspect.signature(LLM.__init__), flush=True)
        except Exception:
            pass

    results: list[SmokeResult] = []
    for spec in specs:
        result = _run_one(spec, args)
        results.append(result)
        if not result.ok and not args.continue_on_error:
            break

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")

    failed = [result for result in results if not result.ok]
    print("\n=== summary ===")
    for result in results:
        status = "OK" if result.ok else "FAIL"
        timing = f" load={result.load_seconds:.2f}s" if result.load_seconds is not None else ""
        print(f"{status:4s} {result.label}{timing}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
