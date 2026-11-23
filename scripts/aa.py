#!/usr/bin/env python3
"""A/A noise floor: how much does the SAME config move from run to run, and how often would the
'beats both neighbouring controls' rule keep a change that does nothing? Standard library only.

  python3 "$S/scripts/bench.py" run ctl.json ctl.json ctl.json ctl.json ctl.json ctl.json ctl.json ctl.json ctl.json ctl.json
  python3 "$S/scripts/aa.py" ledger.jsonl --name ctl --metric c1_output_tps

--name verifies an exact control name. Supply a dedicated ledger of at least ten consecutive
successful runs. Use --lower for a lower-is-better metric such as latency.

With nothing changed, the middle run of any three is the fastest about one time in three, so 'beats both
controls' on its own keeps about a third of useless changes. The floor printed here is the largest win a
no-change run showed over its better neighbour: require a candidate to beat its better control by more.
"""
import json, math, sys


def quantile(v, p):
    s = sorted(v); k = (len(s) - 1) * p; f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def analyse(vals):
    """vals: one metric from consecutive runs of an unchanged config."""
    n = len(vals)
    med = quantile(vals, 0.5)
    mid = range(1, n - 1)
    wins = [i for i in mid if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]]
    over = [vals[i] / max(vals[i - 1], vals[i + 1]) - 1 for i in mid]
    return {"runs": n, "median": med, "min": min(vals), "max": max(vals),
            "spread_p10_p90_pct": 100 * (quantile(vals, 0.9) - quantile(vals, 0.1)) / med,
            "triplets": len(mid), "false_wins": len(wins), "false_win_runs": [i + 1 for i in wins],
            "floor_pct": 100 * max(0, max(over)) if over else None}


def main(argv):
    opt = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d
    flagged = {i for k in ("--name", "--metric") if k in argv for i in (argv.index(k), argv.index(k) + 1)}
    pos = [a for i, a in enumerate(argv) if i not in flagged and a != "--lower"]
    path, name, metric = (pos[0] if pos else "ledger.jsonl"), opt("--name", ""), opt("--metric", "c1_output_tps")
    with open(path, encoding="utf-8") as ledger:
        rows = [json.loads(l) for l in ledger if l.strip()]
    if len(rows) < 10 or any(not r.get("ok") or (name and r.get("name") != name) or
                             not isinstance(r.get(metric), (int, float)) or not math.isfinite(r[metric])
                             or r[metric] <= 0 for r in rows):
        sys.exit("supply a dedicated ledger of >=10 consecutive valid positive measurements of the exact control")
    ids = [r.get("run_id") for r in rows]
    if not all(ids) or len(set(ids)) != len(rows):
        sys.exit("missing or repeated run IDs")
    if not rows[0].get("config_sha256") or len({(r.get("name"), r.get("config_sha256")) for r in rows}) != 1:
        sys.exit("mixed control configurations")
    vals = [r[metric] for r in rows]
    lower = "--lower" in argv
    a = analyse(vals)
    if lower:
        improvements = [1 - vals[i] / min(vals[i-1], vals[i+1]) for i in range(1, len(vals)-1)]
        a.update(floor_pct=100*max(0, max(improvements)), false_wins=sum(x > 0 for x in improvements),
                 false_win_runs=[i+2 for i,x in enumerate(improvements) if x > 0])
    print(f"{a['runs']} runs of the same config, {metric}: median {a['median']:.2f} "
          f"(min {a['min']:.2f}, max {a['max']:.2f}, p10-p90 spread {a['spread_p10_p90_pct']:.1f}%)")
    print(f"no-change runs that beat both neighbours: {a['false_wins']} of {a['triplets']} "
          f"({100 * a['false_wins'] / a['triplets']:.0f}%; chance alone gives about 33%)")
    print(f"noise floor (largest no-change win over the better neighbour): {a['floor_pct']:.2f}%")
    print(f"=> accept a candidate only if it beats its better control by more than {math.ceil(a['floor_pct'] * 10) / 10:.1f}%, "
          f"and repeat any win that is close to that")
    return a


if __name__ == "__main__":
    if "-h" in sys.argv or "--help" in sys.argv:
        sys.exit(__doc__)
    main(sys.argv[1:])
