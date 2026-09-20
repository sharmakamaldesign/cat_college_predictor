"""Command-line entry point.

Usage:
    python -m iim_call_predictor.cli --college iima --input candidate.json
    python -m iim_call_predictor.cli --college iima --input candidate.json --json
    python -m iim_call_predictor.cli --college iima --input candidate.json --mode pool --pool-csv pool.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from . import colleges  # noqa: F401  (import triggers registration of all colleges)
    from .core.models import CandidateInput
    from .core.registry import get_college, list_colleges
except ImportError:
    # Allow running as a plain script (e.g. `python iim_call_predictor/cli.py ...`)
    # in addition to `python -m iim_call_predictor.cli ...`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from iim_call_predictor import colleges  # noqa: F401
    from iim_call_predictor.core.models import CandidateInput
    from iim_call_predictor.core.registry import get_college, list_colleges


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute an IIM shortlisting score for a candidate.")
    parser.add_argument("--college", required=True, help=f"College code. Available: {', '.join(list_colleges())}")
    parser.add_argument("--input", required=True, type=Path, help="Path to a candidate JSON file.")
    parser.add_argument(
        "--mode",
        choices=["reference", "pool"],
        default="reference",
        help="'reference' uses static assumed parameters; 'pool' derives them from --pool-csv.",
    )
    parser.add_argument("--pool-csv", type=Path, help="Path to a candidate-pool CSV, required when --mode pool.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text breakdown.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    college = get_college(args.college)

    candidate_data: Dict[str, Any] = json.loads(args.input.read_text(encoding="utf-8"))
    candidate = CandidateInput(**candidate_data)

    if args.mode == "pool":
        if not args.pool_csv:
            print("error: --pool-csv is required when --mode pool", file=sys.stderr)
            return 2
        params = college.compute_pool_params(args.pool_csv)
    else:
        params = college.load_reference_params()

    result = college.compute_score(candidate, params=params, mode=params["mode"])

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_breakdown(college.name, result)
    return 0


def _print_breakdown(college_name: str, result) -> None:  # noqa: ANN001
    print(f"=== {college_name.upper()} shortlisting score ===")
    print(f"Mode: {result.mode}")
    print()

    print(f"Basic eligibility: {'PASS' if result.eligible else 'FAIL'}")
    for reason in result.eligibility_reasons:
        print(f"  - {reason}")
    print()

    print(f"CAT cutoffs: {'PASS' if result.cutoff_passed else 'FAIL'}")
    for reason in result.cutoff_reasons:
        print(f"  - {reason}")
    print()

    print("Rating scores:")
    for key in ("A", "B", "C", "D", "E"):
        print(f"  {key} = {result.rating_scores[key]}")
    print(f"  Raw AR (A+B+C+D+E) = {result.raw_ar}")
    print()

    print(f"Normalized AR = {result.normalized_ar:.6f}")
    print(f"Raw composite = {result.raw_composite:.6f}")
    print(f"Discipline used = {result.discipline_used}")
    print(f"NCS (Normalized Composite Score) = {result.ncs:.6f}")
    print()

    print(f"Call: {'TRUE' if result.call else 'FALSE'}")
    for reason in result.call_reasons:
        print(f"  - {reason}")
    print()

    print("Params used:")
    for key, value in result.params_used.items():
        print(f"  {key}: {value}")
    print()

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
            
    output = {"eligible": result.eligible, "call": result.call, "ncs": result.ncs}
    print(json.dumps(output))
    return output

if __name__ == "__main__":
    raise SystemExit(main())
