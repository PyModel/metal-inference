"""Offline qualification of one predeclared metric; never changes live settings.

Input: {"metric":"warm_prefix_ttft_s","direction":"lower","noise_pct":3,
"trials":[{"run_ids":["a1","b1","a2"],"aba":[1,0.70,1.01],"checks_passed":true},
          {"run_ids":["a3","b2","a4"],"aba":[1,0.72,1.02],"checks_passed":true}]}
IDs refer to retained native-harness runs.
checks_passed summarizes parity, engagement, protected metrics and resource gates;
retain the underlying evidence. noise_pct comes from matched A/A controls.
"""
import argparse
import json
import math


def positive(x):
    return type(x) in (int, float) and math.isfinite(x) and x > 0


def qualifies(record):
    direction, floor, trials = (record[k] for k in ("direction", "noise_pct", "trials"))
    if (not record["metric"] or direction not in ("lower", "higher")
            or not (positive(floor) or (type(floor) in (int, float) and floor == 0))
            or len(trials) < 2):
        return False
    seen = set()
    for trial in trials:
        ids = trial["run_ids"]
        if (trial["checks_passed"] is not True or len(ids) != 3
                or not all(isinstance(i, str) and i.strip() for i in ids)
                or len(set(ids)) != 3 or seen.intersection(ids)):
            return False
        seen.update(ids)
        left, candidate, right = trial["aba"]
        if not all(positive(x) for x in (left, candidate, right)):
            return False
        best = (max if direction == "higher" else min)(left, right)
        gain = candidate - best if direction == "higher" else best - candidate
        if gain <= max(abs(left - right), best * floor / 100):
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=argparse.FileType("r"))
    args = parser.parse_args()
    data = json.load(args.record)
    passed = qualifies(data)
    print(data["metric"], "qualified" if passed else "retain baseline")
    raise SystemExit(0 if passed else 1)
