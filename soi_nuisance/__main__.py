import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SOI nuisance experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    gui = sub.add_parser("gui", help="Окно параметров, запуска и прогресса")
    gui.add_argument("--preset", default="experiment")
    gui.add_argument("--config")
    for name in ["run", "plan", "estimate"]:
        p = sub.add_parser(name)
        p.add_argument("--preset", choices=["experiment", "smoke", "pilot", "paper", "full", "gaussian_control"], default="experiment")
        p.add_argument("--config")
        p.add_argument("--set", dest="overrides", action="append", default=[], metavar="path=value")
        if name == "run":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--retry-failed", action="store_true",
                           help="Explicitly retry unchanged failed runs from last good checkpoint")
            p.add_argument("--dry-run", action="store_true")
        if name == "estimate":
            p.add_argument("--from-results", required=True)
    p = sub.add_parser("report")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epsilon", type=float)
    args = parser.parse_args()
    if args.command == "gui":
        from .gui import main as gui_main
        gui_main(["--preset", args.preset]+(["--config", args.config] if args.config else []))
        return
    from .config import load_config
    from .runner import run, plan
    from .report import make_report, estimate_duration
    if args.command == "report":
        result = make_report(args.output_dir, args.epsilon)
    else:
        cfg = load_config(args.preset, args.config, args.overrides)
        if args.command == "run":
            cfg["resume"] = args.resume or cfg["resume"]
            cfg["dry_run"] = args.dry_run or cfg["dry_run"]
        if args.command == "plan" or (getattr(args, "dry_run", False) or cfg["dry_run"]):
            result = plan(cfg)
        elif args.command == "estimate":
            result = estimate_duration(cfg, args.from_results)
        else:
            run(cfg, args.resume or cfg["resume"], args.retry_failed)
            result = make_report(cfg["output_dir"], cfg["epsilon"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
