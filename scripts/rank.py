"""Rank independent Metal experiments by estimated recoverable request time."""
import argparse
import json
import math


def rank(profile):
    def ms(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("timings must be nonnegative finite numbers")
        return value

    total = ms(profile["request_ms"])
    if not total:
        raise ValueError("request_ms must be positive")
    rows = []
    for candidate in profile["candidates"]:
        stage = ms(candidate["critical_path_ms"])
        if stage > total:
            raise ValueError("critical path stage exceeds request time")
        saved = stage - ms(candidate["replacement_ms"]) - ms(candidate["handoff_ms"])
        rows.append((candidate["name"], saved, 100 * saved / total))
    return sorted(rows, key=lambda row: -row[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=argparse.FileType("r"))
    args = parser.parse_args()
    try:
        for name, saved, percent in rank(json.load(args.profile)):
            print(f"{saved:+.2f} ms  {percent:+.1f}% latency reduction  {name}")
    except (KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"invalid profile: {error}\n")
