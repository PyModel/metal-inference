"""Decide one metric from two independent native-harness A/B/A summaries."""
import argparse
import json
import math


def qualifies(trials, direction):
    if len(trials) < 2:
        return False
    for trial in trials:
        if trial["checks_passed"] is not True:
            return False
        left, candidate, right = trial["aba"]
        if any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0
               for x in (left, candidate, right)):
            return False
        gain = (candidate - max(left, right) if direction == "higher"
                else min(left, right) - candidate)
        if gain <= abs(left - right):
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("higher", "lower"))
    parser.add_argument("trials", type=argparse.FileType("r"))
    args = parser.parse_args()
    passed = qualifies(json.load(args.trials), args.direction)
    print("qualified" if passed else "retain baseline")
    raise SystemExit(0 if passed else 1)
