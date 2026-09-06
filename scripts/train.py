from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirage training stub")
    parser.add_argument("--data-path", default="data")
    args = parser.parse_args()
    print(f"training stub: {args.data_path}")


if __name__ == "__main__":
    main()
