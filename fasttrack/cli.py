"""Command line. Every setting and folder lives in the experiment file; the command line only names it.

    fasttrack gui [experiment.yaml]      open the app (optionally with a file)
    fasttrack run experiment.yaml        track from raw data, then draw the figures
    fasttrack figures experiment.yaml    redraw the figures from existing results
    fasttrack dynamics experiment.yaml [fit name ...]
                                         fit the stochastic models listed under analysis.dynamics
                                         (all enabled ones, or the named ones) and save their reports
    fasttrack fates experiment.yaml [name ...]
                                         run the fate analyses listed under analysis.fates (committor,
                                         commitment points, features around commitment or division)
    fasttrack export experiment.yaml [name ...]
                                         draw the multi-panel figures listed under figures.exports at
                                         journal size, with captions and the numbers behind each panel
"""
from __future__ import annotations

import sys

USAGE = __doc__.split("\n\n", 1)[1]


def _print_event(event: dict) -> None:
    kind = event["kind"]
    if kind == "batch":
        stages = " ".join(f"{k}={v:.2f}" for k, v in event["timings"].items())
        print(f"  {event['timestack']} {event['frames_done']}/{event['frames_total']} frames: {event['cells']} cells | {stages}")
    elif kind == "timestack_done":
        print(f"{event['id']}: {event['frames']} frames, {event['tracks']} tracks, {event['divisions']} divisions, "
              f"{event['seconds']:.1f}s (waited {event['wait_io']:.1f}s on disk), gate {event['max_travel']:.1f} px, batches {event['batch_sizes']}")
    elif kind == "timestack_skipped":
        print(f"{event['id']}: already processed, skipping")
    elif kind == "timestack_start":
        for problem in event.get("problems", []):
            print(f"{event['id']}: {problem}")
    elif kind == "timestack_failed":
        pass  # reported as a warning by run_experiment
    elif kind in ("warning", "figure_failed"):
        print(f"WARNING {event.get('name', '')} {event['message']}".replace("  ", " "))
    elif kind == "quality_done":
        flagged = event["flagged"]
        print(f"quality check: {len(flagged)} of {event['positions']} positions flagged and skipped (keep them with quality.keep)")
        for key, reason in flagged.items():
            print(f"  {key}: {reason}")
    elif kind == "quality_progress" and event["done"] % 25 == 0:
        print(f"  quality check {event['done']}/{event['total']}")
    elif kind == "figure_done":
        print(f"figure {event['name']}")
    elif kind == "export_note":
        print(f"NOTE {event['name']}: {event['message']}")
    elif kind == "export_done":
        print(f"exported {event['name']}")


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help", "help") or args[0] not in ("gui", "run", "figures", "dynamics", "fates", "export"):
        print(USAGE)
        sys.exit(0 if args and args[0] in ("-h", "--help", "help") else 2)
    command, rest = args[0], args[1:]
    if command == "gui":
        from .gui.app import main as gui_main
        gui_main(rest[0] if rest else None)
        return
    if command == "dynamics" and rest:
        _dynamics(rest[0], rest[1:])
        return
    if command == "fates" and rest:
        _fates(rest[0], rest[1:])
        return
    if command == "export" and rest:
        _export(rest[0], rest[1:])
        return
    if len(rest) != 1:
        print(f"fasttrack {command} takes exactly one experiment file.\n\n{USAGE}")
        sys.exit(2)
    from .experiment import load_experiment
    exp = load_experiment(rest[0])
    if command == "run":
        from .track.run import run_experiment
        run_experiment(exp, _print_event)
    from .analysis.figures import render_all
    written = render_all(exp, _print_event)
    print(f"{len(written)} figure files in {exp['paths']['results']}/{exp['figures']['folder']}")


def _export(path: str, names: list[str]) -> None:
    from .experiment import load_experiment
    from .plotting.export import write_all
    exp = load_experiment(path)
    known = {e["name"] for e in exp["figures"]["exports"]}
    missing = [n for n in names if n not in known]
    if missing:
        print(f"no figure definition called {', '.join(missing)} (have: {', '.join(sorted(known)) or 'none'})")
        sys.exit(2)
    written = write_all(exp, names or None, _print_event)
    print(f"{len(written)} files in {exp['paths']['results']}/{exp['figures']['export_folder']}")


def _dynamics(path: str, names: list[str]) -> None:
    from .dynamics.fit import compare, run_fit
    from .experiment import load_experiment
    exp = load_experiment(path)
    fits = exp["analysis"]["dynamics"]["fits"]
    chosen = [f for f in fits if f["name"] in names] if names else [f for f in fits if f["enabled"]]
    missing = set(names) - {f["name"] for f in fits}
    if missing or not chosen:
        print(f"No fits called {sorted(missing)}; fits are {[f['name'] for f in fits]}" if missing else "No enabled fits in the file.")
        sys.exit(2)

    def show(event):
        if event["type"] == "log" or (event["type"] == "progress" and event["done"] in (1, event["total"])):
            print(event.get("message") or f"  {event['stage']} {event['done']}/{event['total']}", flush=True)

    for fit in chosen:
        report = run_fit(exp, fit, show)
        print(f"{fit['name']}: saved in {report['folder']}")
    for row in compare(exp):
        print(f"  {row['name']:24s} {row['model']:8s} log-likelihood {row['log_likelihood']:12.1f}  BIC {row['bic']:12.1f}")



def _fates(path: str, names: list[str]) -> None:
    from .analysis.fate import fate_folder, run_fate
    from .experiment import load_experiment
    exp = load_experiment(path)
    specs = exp["analysis"]["fates"]
    chosen = [f for f in specs if f["name"] in names] if names else [f for f in specs if f["enabled"]]
    missing = set(names) - {f["name"] for f in specs}
    if missing or not chosen:
        print(f"No fate analyses called {sorted(missing)}; there are {[f['name'] for f in specs]}" if missing
              else "No enabled fate analyses in the file.")
        sys.exit(2)

    def show(event):
        if event["type"] == "log" or (event["type"] == "progress" and event["done"] in (1, event["total"])):
            print(event.get("message") or f"  {event['stage']} {event['done']}/{event['total']}", flush=True)

    for spec in chosen:
        r = run_fate(exp, spec, show)
        print(f"{spec['name']}: threshold {r['threshold']:.3g}, saved in {fate_folder(exp, spec['name'])}")
        for c, name in enumerate(r["conditions"]):
            pts = ", ".join("–" if v is None or v != v else f"{v:.3g}" for v in r["x50"][c])
            print(f"  {name:20s} {r['events'][c]:6d} {r['align']} events  commitment point(s) {pts}")


if __name__ == "__main__":
    main()
