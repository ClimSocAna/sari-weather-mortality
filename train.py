#!/usr/bin/env python
"""Train one native SARI-T member, or resume its epoch checkpoint."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from dataclasses import replace


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_YAML = PROJECT_ROOT / "configs" / "training.example.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runtime" / "training" / "experiments"


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--config", type=Path, default=DEFAULT_CONFIG_YAML,
                      help=f"RunConfig YAML; default: {DEFAULT_CONFIG_YAML}")
    mode.add_argument("--resume", type=Path, help="last_NNN.pt checkpoint with full training state")
    parser.add_argument("--run-directory", type=Path,
                        help=f"Parent of descriptive run folders; defaults to {DEFAULT_OUTPUT_ROOT}; resume uses its saved folder")
    parser.add_argument("--member-id", "--version", dest="member_id", type=int,
                        help="Externally supplied ensemble number; defaults to 1 for new training")
    parser.add_argument("--setup", help="Named experiment override in the input YAML")
    parser.add_argument("--reference", help="Named reference policy in the input YAML")
    parser.add_argument("--kernel-t-days", type=int, help="Override the temperature kernel length in days")
    listing = parser.add_mutually_exclusive_group()
    listing.add_argument("--list-setups", action="store_true", help="List YAML setup names without training")
    listing.add_argument("--list-references", action="store_true", help="List YAML reference names without training")
    listing.add_argument("--print-run-directory", action="store_true", help="Print the resolved descriptive run path without training")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    list_profiles = args.list_setups or args.list_references
    if args.kernel_t_days is not None:
        if args.kernel_t_days < 1:
            parser.error("--kernel-t-days must be a positive integer")
        if args.resume is not None or list_profiles:
            parser.error("--kernel-t-days applies to a new run or --print-run-directory")
    if args.resume is not None and args.member_id is not None:
        parser.error("--resume restores the member number; omit --member-id/--version")
    if args.resume is not None and (args.setup is not None or args.reference is not None or list_profiles or args.print_run_directory):
        parser.error("Setup/reference selection requires --config; --resume restores saved settings")
    if list_profiles:
        import yaml

        values = yaml.safe_load(args.config.expanduser().read_text(encoding="utf-8"))
        group = "setups" if args.list_setups else "references"
        for name in values.get(group, {}):
            print(name)
        return
    if sys.version_info < (3, 10):
        parser.error("Native training requires Python 3.10 or newer")

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from sari_t.run.checkpoint import load_run_config

    run_directory = args.run_directory
    if run_directory is None and args.resume is None:
        run_directory = DEFAULT_OUTPUT_ROOT
    if run_directory is not None:
        run_directory = run_directory.expanduser().resolve()
    if args.resume is None:
        config = load_run_config(args.config, setup=args.setup, reference=args.reference)
        if args.kernel_t_days is not None:
            config = replace(
                config,
                model_config=replace(config.model_config, kernel_T_days=args.kernel_t_days),
            )
        run_directory = run_directory / config.run_name
        if args.print_run_directory:
            print(run_directory)
            return

    import torch

    from sari_t.training.trainer import fit, resume_training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device
    if args.resume is not None:
        checkpoint = args.resume.expanduser().resolve()
        print(f"Resuming {checkpoint} on {device}", flush=True)
        result = resume_training(checkpoint, run_directory=run_directory, device=device)
    else:
        member_id = 1 if args.member_id is None else args.member_id
        print(f"Training member {member_id:03d} in {run_directory} on {device}", flush=True)
        result = fit(config, run_directory, member_id=member_id, device=device)
    if result is None:
        print("Skipping completed member: final checkpoint already exists.", flush=True)
        return
    print(f"Training finished: {result.checkpoint_path}", flush=True)
    print(f"Metrics: {result.metrics_path}", flush=True)


if __name__ == "__main__":
    main()
