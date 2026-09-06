from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirage inference stub")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()
    print(f"inference stub: {args.prompt}")


if __name__ == "__main__":
    main()
