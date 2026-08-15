"""Direct entry point for the selective M04 free-boundary screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from campaign import m04


def run(config_path: str = "configs/independent_m04_free_boundary.yaml", *, output_root: str | Path | None = None, smoke: bool = False) -> dict[str, Any]:
    """Run M04 only through its frozen campaign implementation."""
    return m04.run_m04_screen(config_path=config_path, output_root=output_root, smoke=smoke)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/independent_m04_free_boundary.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, output_root=args.output_root, smoke=args.smoke), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
