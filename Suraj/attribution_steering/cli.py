from __future__ import annotations

import argparse
import json

from .experiment import analyze_collection, collect_dataset, evaluate_steering


def _add_shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None, help="Override device, for example cpu or cuda.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Maximum number of generated answer tokens.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of dataset examples to process.",
    )
    parser.add_argument(
        "--conditions",
        default="clean,misleading",
        help="Comma-separated prompt conditions to run.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiments for attribution-graph analysis and activation steering."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Run baseline generations and cache traces.")
    collect_parser.add_argument("--model", required=True, help="Hugging Face causal LM name or path.")
    collect_parser.add_argument("--dataset", required=True, help="Path to a JSONL factual QA dataset.")
    collect_parser.add_argument("--output-dir", required=True, help="Directory for cached records.")
    _add_shared_runtime_args(collect_parser)

    analyze_parser = subparsers.add_parser("analyze", help="Fit steering vectors from collected traces.")
    analyze_parser.add_argument("--input-dir", required=True, help="Directory produced by collect.")
    analyze_parser.add_argument("--output-dir", required=True, help="Directory for analysis artifacts.")
    analyze_parser.add_argument(
        "--fit-conditions",
        default="misleading",
        help="Comma-separated conditions used to fit the steering state.",
    )
    analyze_parser.add_argument(
        "--top-k-layers",
        type=int,
        default=6,
        help="How many layers to steer.",
    )

    steer_parser = subparsers.add_parser("steer", help="Compare baseline and steered generations.")
    steer_parser.add_argument("--model", required=True, help="Hugging Face causal LM name or path.")
    steer_parser.add_argument("--dataset", required=True, help="Path to a JSONL factual QA dataset.")
    steer_parser.add_argument(
        "--steering-state",
        required=True,
        help="Path to steering_state.pt produced by analyze.",
    )
    steer_parser.add_argument("--output-dir", required=True, help="Directory for steering results.")
    steer_parser.add_argument(
        "--steering-scale",
        type=float,
        default=2.0,
        help="Multiplier applied to steering vectors.",
    )
    _add_shared_runtime_args(steer_parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        result = collect_dataset(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_examples=args.max_examples,
            conditions=args.conditions,
        )
    elif args.command == "analyze":
        result = analyze_collection(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            fit_conditions=args.fit_conditions,
            top_k_layers=args.top_k_layers,
        )
    else:
        result = evaluate_steering(
            model_name=args.model,
            dataset_path=args.dataset,
            steering_state_path=args.steering_state,
            output_dir=args.output_dir,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_examples=args.max_examples,
            conditions=args.conditions,
            steering_scale=args.steering_scale,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
