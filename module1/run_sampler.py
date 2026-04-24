"""
run_sampler.py
--------------
CLI entry point for the Multi-Attribute Sampler.

Usage
-----
    python run_sampler.py                          # uses default INPUT path
    python run_sampler.py path/to/input.json       # custom input file
    python run_sampler.py input.json --variants 6  # request 6 variants
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sampler import MultiAttributeSampler

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT         = "./input/attributes_20260221_172838.json"
DEFAULT_OUTPUT        = "./output"
DEFAULT_ATTRIBUTES_CSV = "./models/bert_celeba/attributes.csv"
DEFAULT_VARIANTS       = 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diverse facial-attribute vectors from BERT predictions."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Path to input JSON file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output base directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--attributes-csv", "-a",
        default=DEFAULT_ATTRIBUTES_CSV,
        dest="attributes_csv",
        help=f"Path to attributes CSV (default: {DEFAULT_ATTRIBUTES_CSV})",
    )
    parser.add_argument(
        "--variants", "-n",
        type=int,
        default=DEFAULT_VARIANTS,
        dest="variants",
        help=f"Number of attribute variants to generate (default: {DEFAULT_VARIANTS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    predicted   = data.get("predicted_attributes", {})
    text_matched = data.get("text_matched_attributes", [])
    input_text  = data.get("input_text", "")

    if not predicted:
        print("[ERROR] 'predicted_attributes' is empty or missing.", file=sys.stderr)
        sys.exit(1)

    sampler = MultiAttributeSampler(args.attributes_csv)

    vectors = sampler.sample_vectors(
        predicted,
        text_matched,
        input_text=input_text,
        num_variants=args.variants,
    )

    folder = sampler.save_tensors(vectors, args.output)

    print(f"\n✓ {len(vectors)} variant(s) generated and saved to: {folder}\n")
    print("=" * 60)

    for i, vec in enumerate(vectors):
        description = sampler.generate_description(vec)
        print(f"  Variant {i + 1}: {description}")

    print("=" * 60)


if __name__ == "__main__":
    main()