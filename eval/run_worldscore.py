from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="WorldScore evaluation stub")
    parser.add_argument("--input", default="")
    args = parser.parse_args()
    print(f"worldscore stub: {args.input}")


if __name__ == "__main__":
    main()
