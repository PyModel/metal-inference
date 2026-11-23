"""CPU-only startup check. Run in the actual trial directory and isolated env."""
import argparse
from pathlib import Path
import subprocess


def probe(binary):
    binary = str(Path(binary).resolve(strict=True))
    for flag in ("--version", "--help"):
        subprocess.run([binary, flag], check=True, timeout=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    probe(parser.parse_args().binary)
    print("binary starts; settings and model serving not tested")
