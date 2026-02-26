#!/usr/bin/env python3
import argparse
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


DEFAULT_METHODS = ["GradAscent", "GradDiff", "NPO", "SimNPO", "RMU", "DPO"]
ALT_REQUIRED_METHODS = {"DPO"}


def _find_free_port() -> int:
    try:
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]
    except OSError:
        # Some sandboxed environments disallow binding ephemeral ports.
        return 29500


def _parse_methods(values: Iterable[str]) -> List[str]:
    methods: List[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                methods.append(item)
    return methods


def _build_common_overrides(args: argparse.Namespace) -> List[str]:
    overrides = [
        "trainer.args.do_train=True",
        f"trainer.args.do_eval={'True' if args.enable_eval else 'False'}",
    ]
    if not args.enable_eval:
        overrides.append("eval=null")

    if args.model is not None:
        overrides.append(f"model={args.model}")
    if args.model_path is not None:
        overrides.append(
            f"model.model_args.pretrained_model_name_or_path={args.model_path}"
        )
    if args.num_train_epochs is not None:
        overrides.append(f"trainer.args.num_train_epochs={args.num_train_epochs}")
    if args.learning_rate is not None:
        overrides.append(f"trainer.args.learning_rate={args.learning_rate}")
    if args.per_device_train_batch_size is not None:
        overrides.append(
            f"trainer.args.per_device_train_batch_size={args.per_device_train_batch_size}"
        )
    if args.gradient_accumulation_steps is not None:
        overrides.append(
            f"trainer.args.gradient_accumulation_steps={args.gradient_accumulation_steps}"
        )
    return overrides


def _quote_cmd(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _build_post_eval_cmd(
    args: argparse.Namespace, method: str, task_name: str
) -> List[str]:
    model_path = args.post_eval_model_path_template.format(
        task_name=task_name, method=method
    )
    cmd = [
        "python",
        "src/eval.py",
        f"task_name={task_name}",
        f"model={args.model}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
    ]
    if args.post_eval_experiment is not None:
        cmd.append(f"experiment={args.post_eval_experiment}")
    cmd.extend(args.post_eval_overrides)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multiple unlearning methods on local data/unlearn JSONL files."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help=(
            "Methods to run. Accepts space-separated names and/or comma-separated values. "
            f"Default: {', '.join(DEFAULT_METHODS)}"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/unlearn",
        help="Directory containing forget.jsonl, retain.jsonl, forget_alt.jsonl.",
    )
    parser.add_argument(
        "--model",
        default="Llama-3.2-1B-Instruct",
        help="Hydra model config name (configs/model/*.yaml).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model.model_args.pretrained_model_name_or_path.",
    )
    parser.add_argument(
        "--task-prefix",
        default="json",
        help="Prefix for task_name. Final names look like <task-prefix>_<method>.",
    )
    parser.add_argument(
        "--accelerate-config",
        default="configs/accelerate/default_config.yaml",
        help="Path to accelerate config file.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value (e.g. '0,1').",
    )
    parser.add_argument(
        "--main-process-port",
        type=int,
        default=None,
        help="Accelerate main process port. If omitted, auto-picks a free port.",
    )
    parser.add_argument(
        "--enable-eval",
        action="store_true",
        help="Enable eval during training (disabled by default for local JSON unlearn runs).",
    )
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--post-eval",
        action="store_true",
        help="Run src/eval.py after each successful training run.",
    )
    parser.add_argument(
        "--post-eval-experiment",
        default=None,
        help=(
            "Optional eval experiment override (e.g., eval/tofu/default.yaml or "
            "eval/muse/default.yaml)."
        ),
    )
    parser.add_argument(
        "--post-eval-model-path-template",
        default="saves/unlearn/{task_name}",
        help=(
            "Template for trained model path passed to eval. "
            "Supports {task_name} and {method}."
        ),
    )
    parser.add_argument(
        "--post-eval-override",
        dest="post_eval_overrides",
        action="append",
        default=[],
        help="Additional Hydra override to append to src/eval.py (repeatable).",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help="Additional Hydra overrides appended to every run.",
    )
    args = parser.parse_args()

    methods = _parse_methods(args.methods)
    if not methods:
        raise SystemExit("No methods provided.")

    data_dir = Path(args.data_dir)
    forget_path = data_dir / "forget.jsonl"
    retain_path = data_dir / "retain.jsonl"
    forget_alt_path = data_dir / "forget_alt.jsonl"

    required = [forget_path, retain_path]
    if any(m in ALT_REQUIRED_METHODS for m in methods):
        required.append(forget_alt_path)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required data files: {missing}")

    common_overrides = _build_common_overrides(args)
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for method in methods:
        uses_alt = method in ALT_REQUIRED_METHODS
        forget_cfg = "JSON_QA_forget_alt" if uses_alt else "JSON_QA_forget"
        forget_data_file = forget_alt_path if uses_alt else forget_path
        task_name = f"{args.task_prefix}_{method}"
        port = args.main_process_port or _find_free_port()

        hydra_overrides = [
            "trainer.args.remove_unused_columns=False",
            "data/datasets@data.forget=" + forget_cfg,
            "data/datasets@data.retain=JSON_QA_retain",
            f"trainer={method}",
            f"task_name={task_name}",
            f"data.forget.{forget_cfg}.args.hf_args.data_files={forget_data_file.as_posix()}",
            f"data.retain.JSON_QA_retain.args.hf_args.data_files={retain_path.as_posix()}",
        ]

        cmd = [
            "accelerate",
            "launch",
            "--config_file",
            args.accelerate_config,
            "--main_process_port",
            str(port),
            "src/train.py",
            "--config-name=unlearn.yaml",
            *hydra_overrides,
            *common_overrides,
            *args.hydra_overrides,
        ]

        print(f"\n[{method}] {'(uses forget_alt)' if uses_alt else ''}".rstrip())
        print(_quote_cmd(cmd))
        if args.post_eval:
            eval_cmd = _build_post_eval_cmd(args=args, method=method, task_name=task_name)
            print(f"[{method}] post-eval")
            print(_quote_cmd(eval_cmd))
        if args.dry_run:
            continue

        completed = subprocess.run(cmd, env=env)
        if completed.returncode != 0:
            return completed.returncode

        if args.post_eval:
            eval_completed = subprocess.run(eval_cmd, env=env)
            if eval_completed.returncode != 0:
                return eval_completed.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
