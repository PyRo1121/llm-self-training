"""Register a completed run from disk into the control plane warehouse."""

from __future__ import annotations

import argparse
import sys

from llm_core.register_run import register_run_from_disk


def main() -> None:
    parser = argparse.ArgumentParser(description="Register training run in warehouse")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--status", default="completed")
    args = parser.parse_args()

    try:
        out = register_run_from_disk(args.run_name, status=args.status)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(
        f"Registered {out['run_id']} status={out['status']} "
        f"adapter={out.get('adapter_path')}"
    )


if __name__ == "__main__":
    main()
