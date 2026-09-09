"""
Read the `levels` blocks out of a pre-encoded dataset and report what any silence filter
would have kept.

The pre-encode script measures RMS and silence-fraction on every item it sees, but only
writes the sidecar JSON for items that survive the filter (pre_encode_dataset.py:478). So a
filtered pass yields a distribution truncated at its own thresholds: after
`--max_silence_fraction 0.3`, every surviving item is below 0.3 by construction and the
distribution says nothing about where the cutoff should have been. Point this script at a
pass encoded with `--no_silence_filter` and it reports the uncensored distribution plus the
survivor count at every candidate threshold.

  uv run python scripts/analyze_silence_levels.py --dir <output_path>
  uv run python scripts/analyze_silence_levels.py --dir <output_path> --variant v0 \
      --rms_floor -50 --max_silence_fraction 0.3

What it cannot see: items dropped *before* the level check, which write no sidecar at all —
`is_silence` in dataset.py (peak below -60 dBFS over the whole crop) and the metadata fn's
own rejects. `--no_silence_filter` does not switch those off. Their counts are read from
`_skipped.json` and reported as census coverage so the survivor percentages below are
against the real corpus size rather than against the sidecars that happen to exist. Neither
bucket biases a cutoff choice: both hold items no cutoff would have kept.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np

# The two buckets that leave no sidecar, keyed by the `_skipped.json` reason they appear
# under. Everything else in that file is a level-check drop, which a --no_silence_filter
# pass will not contain.
PRE_LEVEL_REASONS = (
    "peak below silence threshold",
    "accompaniment silent over the encoded window",
    "no accompaniment stems",
    "rejected by the metadata fn",
)

DEFAULT_CUTOFFS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_FLOORS = (-70.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0)


def load_items(output_dir, variant=None):
    """Return [(latent_id, variant, levels)] for every sidecar carrying a levels block."""
    items, missing = [], 0
    for path in sorted(glob.glob(os.path.join(output_dir, "[0-9]*.json"))):
        stem = os.path.basename(path)[: -len(".json")]
        m = re.search(r"_v(\d+)$", stem)
        v = f"v{m.group(1)}" if m else "v0"
        if variant is not None and v != variant:
            continue
        with open(path) as f:
            md = json.load(f)
        levels = md.get("levels")
        if not levels:
            missing += 1
            continue
        items.append((stem, v, levels))
    return items, missing


def quantiles(values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    q = np.percentile(finite, [0, 5, 25, 50, 75, 95, 100])
    return {
        "n": int(a.size),
        "n_neg_inf": int(a.size - finite.size),
        "min": q[0], "p05": q[1], "p25": q[2],
        "median": q[3], "p75": q[4], "p95": q[5], "max": q[6],
    }


def survives(levels, floor, cutoff):
    """Replicate the filter in pre_encode_dataset.py: every stream must clear both tests."""
    for level in levels.values():
        if floor is not None and level["rms_dbfs"] < floor:
            return False
        if cutoff is not None and level["silence_fraction"] > cutoff:
            return False
    return True


def first_failure(levels, stream_order, floor, cutoff):
    """The bucket `_skipped.json` would have recorded: first stream, RMS tested before
    fraction. Returns None if the item survives."""
    for name in stream_order:
        level = levels[name]
        if floor is not None and level["rms_dbfs"] < floor:
            return f"{name} below the RMS floor"
        if cutoff is not None and level["silence_fraction"] > cutoff:
            return f"{name} over the silence-fraction limit"
    return None


def report(output_dir, variant, floors, cutoffs, floor, cutoff):
    items, missing = load_items(output_dir, variant)
    if not items:
        raise SystemExit(
            f"No sidecars with a `levels` block under {output_dir}"
            + (f" for variant {variant}" if variant else "")
            + (f" ({missing} sidecar(s) had no levels block — encoded before the filter landed?)"
               if missing else "")
        )

    variants = sorted({v for _, v, _ in items})
    # Stream order matters only for the first-failure attribution: target, then controls in
    # the order the encode wrote them.
    keys = list(items[0][2].keys())
    stream_order = ["target"] + [k for k in keys if k != "target"]

    print(f"Levels census: {output_dir}")
    print(f"  sidecars with levels : {len(items)}"
          f"   variants: {', '.join(variants)}"
          + (f"   (filtered to {variant})" if variant else ""))
    if missing:
        print(f"  sidecars without levels: {missing}  (pre-date the filter; excluded)")
    print(f"  streams              : {', '.join(stream_order)}")

    # Coverage. Anything in _skipped.json under a pre-level reason never reached the level
    # check and so has no levels to census; anything else there means the pass was filtered
    # and the distribution below is truncated.
    corpus = {}
    skipped_path = os.path.join(output_dir, "_skipped.json")
    if os.path.exists(skipped_path):
        with open(skipped_path) as f:
            skip_report = json.load(f)
        print("\nCensus coverage (from _skipped.json):")
        for v in variants:
            entry = skip_report.get(v)
            if not entry:
                print(f"  {v}: not in _skipped.json")
                continue
            pre = {r: n for r, n in entry["skipped"].items() if r in PRE_LEVEL_REASONS}
            level_drops = {r: n for r, n in entry["skipped"].items() if r not in PRE_LEVEL_REASONS}
            total = entry["written"] + sum(entry["skipped"].values())
            corpus[v] = total
            print(f"  {v}: {total} source item(s) -> {entry['written']} written, "
                  f"{sum(entry['skipped'].values())} skipped")
            for r, n in sorted(pre.items(), key=lambda kv: -kv[1]):
                print(f"       {n:>6}  {r}  (no sidecar; never level-checked)")
            if level_drops:
                print("       ** this pass was FILTERED — the distribution below is truncated "
                      "at its own thresholds and must not be used to choose them **")
                for r, n in sorted(level_drops.items(), key=lambda kv: -kv[1]):
                    print(f"       {n:>6}  {r}")
    else:
        print("\nNo _skipped.json — cannot state census coverage; percentages are of sidecars only.")

    # Distribution.
    print("\nDistribution over the census"
          f" ({len(items)} item(s)):")
    header = f"  {'stream':<18}{'measure':<18}" + "".join(
        f"{h:>9}" for h in ("min", "p05", "p25", "median", "p75", "p95", "max"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    dist = {}
    for name in stream_order:
        for measure, fmt in (("rms_dbfs", "{:>9.1f}"), ("silence_fraction", "{:>9.3f}")):
            q = quantiles([lv[name][measure] for _, _, lv in items])
            dist[f"{name}.{measure}"] = q
            row = f"  {name:<18}{measure:<18}" + "".join(
                fmt.format(q[k]) for k in ("min", "p05", "p25", "median", "p75", "p95", "max"))
            print(row + (f"   ({q['n_neg_inf']} at -inf)" if q["n_neg_inf"] else ""))

    n_corpus = sum(corpus.values()) if corpus else len(items)

    # Sweeps. Each holds the other knob at the value passed in, so the two are read as
    # "what does this knob cost me, given the other one".
    print(f"\nsilence-fraction sweep (RMS floor held at "
          f"{'off' if floor is None else f'{floor:g} dBFS'}):")
    print(f"  {'cutoff':>8}{'target only':>14}{'all streams':>14}{'% census':>10}{'% corpus':>10}")
    sweep_fraction = []
    for c in cutoffs:
        t_only = sum(1 for _, _, lv in items
                     if survives({"target": lv["target"]}, floor, c))
        allk = sum(1 for _, _, lv in items if survives(lv, floor, c))
        sweep_fraction.append({"cutoff": c, "target_only": t_only, "all_streams": allk})
        print(f"  {c:>8.2f}{t_only:>14}{allk:>14}"
              f"{100 * allk / len(items):>9.0f}%{100 * allk / n_corpus:>9.0f}%")

    print(f"\nRMS-floor sweep (silence fraction held at "
          f"{'off' if cutoff is None else f'{cutoff:g}'}):")
    print(f"  {'floor':>8}{'target only':>14}{'all streams':>14}{'% census':>10}{'% corpus':>10}")
    sweep_rms = []
    for fl in floors:
        t_only = sum(1 for _, _, lv in items
                     if survives({"target": lv["target"]}, fl, cutoff))
        allk = sum(1 for _, _, lv in items if survives(lv, fl, cutoff))
        sweep_rms.append({"floor": fl, "target_only": t_only, "all_streams": allk})
        print(f"  {fl:>8.0f}{t_only:>14}{allk:>14}"
              f"{100 * allk / len(items):>9.0f}%{100 * allk / n_corpus:>9.0f}%")

    # How the two tests overlap, which the bucket counts in _skipped.json cannot show: an
    # item is recorded under its FIRST failure only, and RMS is tested before fraction, so
    # the fraction bucket is always missing however many low-RMS items would also have
    # failed it.
    print(f"\nOverlap of the two tests at floor "
          f"{'off' if floor is None else f'{floor:g} dBFS'} / cutoff "
          f"{'off' if cutoff is None else f'{cutoff:g}'} (target stream):")
    tab = Counter()
    for _, _, lv in items:
        low_rms = floor is not None and lv["target"]["rms_dbfs"] < floor
        high_frac = cutoff is not None and lv["target"]["silence_fraction"] > cutoff
        tab[(low_rms, high_frac)] += 1
    print(f"  {'':<22}{'frac OK':>12}{'frac over':>12}")
    print(f"  {'RMS OK':<22}{tab[(False, False)]:>12}{tab[(False, True)]:>12}")
    print(f"  {'RMS below floor':<22}{tab[(True, False)]:>12}{tab[(True, True)]:>12}")
    print("  (RMS is tested first, so the whole bottom row is labelled 'below the RMS floor' "
          "in _skipped.json\n   however many of them are also over the fraction limit.)")

    buckets = Counter(first_failure(lv, stream_order, floor, cutoff) for _, _, lv in items)
    kept = buckets.pop(None, 0)
    print(f"\nAt floor {'off' if floor is None else f'{floor:g}'} / cutoff "
          f"{'off' if cutoff is None else f'{cutoff:g}'}, this census would give:")
    print(f"  {kept:>6}  written (of {len(items)} censused, {n_corpus} in the corpus)")
    for r, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>6}  {r}")

    return {
        "dir": output_dir,
        "variant": variant,
        "censused": len(items),
        "corpus_per_variant": corpus,
        "streams": stream_order,
        "distribution": dist,
        "sweep_silence_fraction": sweep_fraction,
        "sweep_rms_floor": sweep_rms,
        "at_candidate": {
            "rms_floor": floor,
            "max_silence_fraction": cutoff,
            "written": kept,
            "first_failure": dict(buckets),
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True,
                   help="Pre-encode output_path holding the .json sidecars")
    p.add_argument("--variant", default=None,
                   help="Only census this variant (e.g. v0). Default: all of them.")
    p.add_argument("--rms_floor", type=float, default=-50.0,
                   help="Candidate --silence_threshold_db to evaluate (default -50). "
                        "Held fixed during the silence-fraction sweep.")
    p.add_argument("--max_silence_fraction", type=float, default=0.3,
                   help="Candidate --max_silence_fraction to evaluate (default 0.3). "
                        "Held fixed during the RMS-floor sweep.")
    p.add_argument("--cutoffs", type=float, nargs="*", default=list(DEFAULT_CUTOFFS),
                   help="silence_fraction values to sweep")
    p.add_argument("--floors", type=float, nargs="*", default=list(DEFAULT_FLOORS),
                   help="RMS floors in dBFS to sweep")
    p.add_argument("--json", default=None, help="Also write the report here as JSON")
    a = p.parse_args()

    out = report(a.dir, a.variant, a.floors, a.cutoffs, a.rms_floor, a.max_silence_fraction)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {a.json}")
