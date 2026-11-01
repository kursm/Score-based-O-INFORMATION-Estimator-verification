"""Background process for the desktop UI. JSON events share stdout with the CLI log."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

EVENT_PREFIX = "@soi-event "


def send(event, **values):
    print(EVENT_PREFIX+json.dumps(dict(event=event, **values), ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["preview", "run", "report"], required=True)
    parser.add_argument("--stop-file", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "report":
            # Rebuild from saved results, independent of draft model parameters in the window.
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
            send("report")
            from .report import make_report
            report = make_report(cfg["output_dir"], cfg.get("epsilon"))
            send("finished", output_dir=str(Path(cfg["output_dir"]).resolve()), report=report,
                 successful=report["statuses"].get("ok", 0), total=sum(report["statuses"].values()))
            return 0
        from .config import ROOT, load_config
        from .runner import plan, run
        from .gui_progress import build_tasks
        import torch
        cfg = load_config(config=args.config)
        device = cfg["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA недоступна. Выберите device=cpu или auto.")
        info = plan(cfg)
        tasks = build_tasks(cfg, info, device, ROOT)
        send("plan", plan=info, tasks=tasks, device=device)
        if args.mode == "preview" or cfg["dry_run"]:
            send("preview_finished")
            return 0
        stop_file = Path(args.stop_file)
        last_check, stopped = 0., False
        def should_stop():
            nonlocal last_check, stopped
            now = time.monotonic()
            if now-last_check >= .2:
                stopped, last_check = stop_file.exists(), now
            return stopped
        def forward(event):
            event = event.copy()
            kind = event.pop("event")
            send(kind, **event)
        try:
            results = run(cfg, resume=cfg["resume"], progress_callback=forward, stop_requested=should_stop)
        except KeyboardInterrupt:
            send("stopped", output_dir=str(Path(cfg["output_dir"]).resolve()))
            return 0
        if stop_file.exists():
            send("stopped", output_dir=str(Path(cfg["output_dir"]).resolve()))
            return 0
        send("report")
        from .report import make_report
        report = make_report(cfg["output_dir"], cfg["epsilon"])
        send("finished", output_dir=str(Path(cfg["output_dir"]).resolve()), report=report,
             successful=sum(r["status"] == "ok" for r in results), total=len(results))
        return 0
    except Exception as exc:
        traceback.print_exc()
        send("error", message=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
