#!/usr/bin/env python3
"""Generate deterministic synthetic retrieval data for Chapter 3."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

PRODUCTS = (
    {"name": "Atlas Gateway", "prefix": "AX4", "kind": "edge gateway", "component": "gateway control plane"},
    {"name": "Nimbus Sensor", "prefix": "NSN", "kind": "environmental sensor", "component": "sensor telemetry service"},
    {"name": "Orion Switch", "prefix": "ORS", "kind": "network switch", "component": "switch management plane"},
    {"name": "Helios Reader", "prefix": "HRD", "kind": "identity reader", "component": "reader authentication service"},
)
DOCUMENT_TYPES = ("product", "policy", "operations", "pricing", "specification")
REGIONS = ("north", "south", "east", "west")
SUPPORT_TIERS = ("Standard", "Priority", "Enterprise")
RETENTION_DAYS = (30, 90, 180, 365, 730)
TEMPERATURE_LIMITS = (-20, 0, 5, 25, 40, 55)
STORAGE_TIERS = ("warm", "cool", "archive")
FAILURE_MODES = ("keyword_exact_match", "semantic_paraphrase", "long_context", "table_lookup")


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


def document_values(index: int, rng: random.Random) -> dict[str, Any]:
    product = dict(rng.choice(PRODUCTS))
    document_type = rng.choice(DOCUMENT_TYPES)
    fault_code = f"{product['prefix']}-{index:04d}"
    return {
        "index": index,
        "doc_id": f"{document_type}_{index:04d}_{fault_code.lower()}",
        "product": product,
        "document_type": document_type,
        "region": rng.choice(REGIONS),
        "support_tier": rng.choice(SUPPORT_TIERS),
        "retention_days": rng.choice(RETENTION_DAYS),
        "temperature_c": rng.choice(TEMPERATURE_LIMITS),
        "incident_code": "INC-" + rng.choice(("P1", "P2", "P3")),
        "change_code": f"CC-{rng.randint(10, 99)}",
        "fault_code": fault_code,
        "storage_tier": rng.choice(STORAGE_TIERS),
    }


def render_document(values: dict[str, Any]) -> str:
    product = values["product"]
    title = (
        f"{product['name']} {values['document_type'].title()} "
        f"Record {values['index']:04d}"
    )
    return f"""# {title}

## Record summary

Document ID: `{values['fault_code']}`  
Document type: {values['document_type']}  
Operating region: {values['region']}  
Primary component: {product['component']}

This synthetic record covers the fictional {product['kind']} identified as `{values['fault_code']}`. The record belongs to the {values['region']} operating unit and is retained for retrieval testing only.

## Operating notes

During the pre-change review, the team recorded change ticket `{values['change_code']}` and classified the related alert as `{values['incident_code']}`. Routine telemetry reviews include duplicate event checks, maintenance-window acknowledgements, and inventory reconciliation. These notes create distractor language around facts needed by a search task.

The identity renewal procedure must begin before the certificate rollover window. For this document, renewal ownership remains with the {product['component']} team. A reviewer should not infer that another product family has the same rotation schedule.

## Service and limits

| Field | Value |
| --- | --- |
| Product | {product['name']} |
| Support plan | {values['support_tier']} |
| Temperature limit | {values['temperature_c']} C |
| Retention period | {values['retention_days']} days |
| Storage tier | {values['storage_tier']} |

## Procedure detail

1. Confirm the asset identifier matches `{values['fault_code']}` before opening a maintenance record.
2. Check the certificate expiry report and assign the identity renewal task to the owning component team.
3. Record any exception against `{values['change_code']}` and use `{values['incident_code']}` only for the stated severity.
4. Preserve the operational record for {values['retention_days']} days in the {values['storage_tier']} tier.

## Scope note

All names, codes, policies, products, and values in this document are fictional. The document contains no customer, employer, or production data.
"""


def evaluation_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for offset, values in enumerate(items[: min(32, len(items))], start=1):
        product = values["product"]
        mode = FAILURE_MODES[(offset - 1) % len(FAILURE_MODES)]
        if mode == "keyword_exact_match":
            query = f"What record contains identifier {values['fault_code']}?"
            notes = "Exact synthetic asset identifier."
        elif mode == "semantic_paraphrase":
            query = f"Which document assigns responsibility for certificate rollover for {product['name']}?"
            notes = "The document uses identity renewal rather than certificate rollover."
        elif mode == "long_context":
            query = f"Which record says how long {product['name']} operational data must be preserved?"
            notes = "Retention detail follows operational distractor text."
        else:
            query = f"What is the temperature limit listed for {product['name']} record {values['fault_code']}?"
            notes = "Exact value appears in a Markdown table."
        records.append(
            {
                "id": f"q-{offset:03d}",
                "query": query,
                "expected_doc_ids": [values["doc_id"]],
                "failure_mode": mode,
                "notes": notes,
            }
        )
    return records


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.documents < 1:
        raise SystemExit("--documents must be at least 1.")

    rng = random.Random(args.seed)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    corpus_dir = args.output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for index in range(1, args.documents + 1):
        values = document_values(index, rng)
        output_path = corpus_dir / f"{values['doc_id']}.md"
        output_path.write_text(render_document(values), encoding="utf-8")
        items.append(values)

    manifest = {
        "documents": args.documents,
        "seed": args.seed,
        "items": [
            {
                "doc_id": item["doc_id"],
                "path": f"corpus/{item['doc_id']}.md",
                "document_type": item["document_type"],
                "failure_tags": list(FAILURE_MODES),
            }
            for item in items
        ],
    }
    records = evaluation_records(items)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "eval_queries.json", records)
    print(f"Generated {args.documents} documents in {args.output_dir}")
    print(f"Wrote {len(records)} evaluation queries")


if __name__ == "__main__":
    main()
