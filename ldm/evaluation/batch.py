"""Shared directory runner: resume, dry-run, plots-only and failure capture.

Every script that accepts ``--input_dir`` routes through :func:`run_batch`, so
they all resume the same way, write per-item subdirectories under ``--outdir``,
and record failures in ``failed_runs.json`` instead of aborting (unless
``--strict``).
"""

import os
import os.path as osp

from ldm.evaluation.report import read_json, write_json

RESULT_FILENAMES = ("metrics.json", "results.json")


def handle_offline(args, discover, replot, single_output=False, summarize=None):
    """Handle dry-run/plots-only before loading checkpoints or creating models."""
    if not (args.dry_run or args.plots_only):
        return False
    if args.input_dir is not None and not single_output:
        _layout, items = discover()
        results, failures = run_batch(
            items, args.outdir, None, dry_run=args.dry_run,
            plots_only=args.plots_only and not args.dry_run,
            replot=replot, strict=args.strict,
        )
        if summarize is not None and args.plots_only and not args.dry_run:
            summarize(results, failures)
    elif args.dry_run:
        print(f"[dry-run] would evaluate {args.input or getattr(args, 'dataset_dir', None) or args.input_dir} "
              f"-> {args.outdir}")
    else:
        path = default_result_path(args.outdir)
        if path is None:
            raise FileNotFoundError(f"No evaluation results in {args.outdir}")
        replot(None, args.outdir, read_json(path))
    return True


def default_result_path(item_outdir):
    """The result document an earlier run of this item would have written."""
    for name in RESULT_FILENAMES:
        path = osp.join(item_outdir, name)
        if osp.isfile(path):
            return path
    return None


def is_complete(item_outdir, required=()):
    """Whether an item already has a result document and every required file."""
    if default_result_path(item_outdir) is None:
        return False
    return all(osp.isfile(osp.join(item_outdir, name)) for name in required)


def run_batch(items, outdir, run_one, kind="item", required_outputs=(),
              replot=None, force=False, dry_run=False, plots_only=False,
              strict=False, complete=None):
    """Run ``run_one(item, item_outdir)`` over ``items``, one subdirectory each.

    ``run_one`` returns a result dict, or None when the item is unusable (too
    short, say), which is recorded as a failure rather than a result.

    Returns (results, failures) where results holds ``(name, result)`` pairs.
    """
    results, failures = [], []
    total = len(items)

    for index, item in enumerate(items, 1):
        name = item.name
        item_outdir = osp.join(outdir, name)
        print(f"\n[{index}/{total}] {name}")

        if plots_only and not dry_run:
            _replot_one(name, item, item_outdir, replot, results, failures, kind, strict)
            continue

        existing = None if force else default_result_path(item_outdir)
        if existing is not None and is_complete(item_outdir, required_outputs) \
                and (complete is None or complete(item, item_outdir)):
            print(f"  [skip] already complete ({osp.basename(existing)})")
            results.append((name, read_json(existing)))
            continue

        if dry_run:
            print(f"  [dry-run] would evaluate {item.path} -> {item_outdir}")
            continue

        try:
            os.makedirs(item_outdir, exist_ok=True)
            result = run_one(item, item_outdir)
        except Exception as exc:  # one bad recording should not sink the batch
            print(f"  FAILED: {exc}")
            failures.append({kind: name, "path": item.path, "error": repr(exc)})
            if strict:
                raise
            continue

        if result is None:
            failures.append({kind: name, "path": item.path, "error": "too_short"})
            continue
        results.append((name, result))

    if failures and not dry_run:
        write_json(failures, outdir, "failed_runs.json",
                   what=f"{len(failures)} failure(s)")
    return results, failures


def _replot_one(name, item, item_outdir, replot, results, failures, kind, strict):
    """Rebuild one item's figures from the result document it already has."""
    existing = default_result_path(item_outdir)
    if existing is None:
        print("  [skip] no existing results to plot from")
        failures.append({kind: name, "path": item.path, "error": "no_results"})
        return
    result = read_json(existing)
    results.append((name, result))
    if replot is None:
        print("  [skip] this script has no plots-only path")
        return
    try:
        replot(item, item_outdir, result)
    except Exception as exc:
        print(f"  FAILED to re-plot: {exc}")
        failures.append({kind: name, "path": item.path, "error": repr(exc)})
        if strict:
            raise


def summarize(results, failures, outdir, extra=None, aggregate=None,
              filename="summary.json", what="directory summary"):
    """Write ``summary.json`` describing a batch run."""
    summary = {
        "outdir": osp.abspath(outdir),
        "n_evaluated": len(results),
        "n_failures": len(failures),
    }
    if extra:
        summary.update(extra)
    if aggregate is not None:
        summary["aggregate"] = aggregate
    summary["items"] = [{"name": name, **_scalarize(result)}
                        for name, result in results]
    summary["failures"] = failures
    write_json(summary, outdir, filename, what=what)
    return summary


def _scalarize(result):
    """Drop bulky per-window arrays so a summary stays readable."""
    skip = {"logits", "softmax", "predictions", "confidence", "window_gt",
            "window_label_details", "runs", "sources", "window_gt_per_source"}
    return {key: value for key, value in result.items() if key not in skip}
