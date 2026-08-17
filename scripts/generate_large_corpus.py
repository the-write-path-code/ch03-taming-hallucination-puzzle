#!/usr/bin/env python3
"""Generate a deterministic synthetic corpus for retrieval experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PRODUCTS = (
    ("Atlas Gateway", "AX4", "gateway"),
    ("Nimbus Sensor", "NSN", "sensor"),
    ("Orion Switch", "ORS", "switch"),
    ("Helios Reader", "HRD", "reader"),
)
DOCUMENT_TYPES = ("product", "policy", "operations", "pricing", "specification")
REGIONS = ("north", "south", "east", "west")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic enterprise retrieval corpus."
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=200,
        help="Number of Markdown documents to generate (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic output (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_data/generated_large"),
        help="Destination directory (default: sample_data/generated_large).",
    )
    return parser.parse_args()


def build_document(index: int, rng: random.Random) -> tuple[str, str]:
    product_name, prefix, product_kind = rng.choice(PRODUCTS)
    document_type = rng.choice(DOCUMENT_TYPES)
    region = rng.choice(REGIONS)
    identifier = f"{prefix}-{index:04d}"
    retention_days = rng.choice((30, 90, 180, 365, 730))
    temperature_c = rng.choice((-20, 0, 5, 25, 40, 55))
    support_tier = rng.choice(("Standard", "Priority", "Enterprise"))
    title = f"{product_name} {document_type.title()} {index:04d}"
    body = f"""# {title}

Document ID: `{identifier}`
Document type: {document_type}
Region: {region}

The fictional {product_kind} record `{identifier}` supports the {region} operating unit.
Identity renewal is scheduled before the certificate rollover window closes.

| Field | Value |
| --- | --- |
| Product | {product_name} |
| Support plan | {support_tier} |
| Temperature limit | {temperature_c} C |
| Retention period | {retention_days} days |

For testing only, this document contains synthetic enterprise content and no production data.
"""
    filename = f"{document_type}_{index:04d}_{identifier.lower()}.md"
    return filename, body


def main() -> None:
    args = parse_args()
    if args.documents < 1:
        raise SystemExit("--documents must be at least 1.")

    rng = random.Random(args.seed)
    corpus_dir = args.output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    for index in range(1, args.documents + 1):
        filename, body = build_document(index, rng)
        (corpus_dir / filename).write_text(body, encoding="utf-8")
        manifest.append({"doc_id": Path(filename).stem, "path": f"corpus/{filename}"})

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {"documents": args.documents, "seed": args.seed, "items": manifest},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Generated {args.documents} documents in {args.output_dir}")


if __name__ == "__main__":
    main()
