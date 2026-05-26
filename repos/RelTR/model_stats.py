import argparse
import json
import sys

import torch

from main import get_args_parser
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        "RelTR parameter/FLOPs reporter",
        parents=[get_args_parser()],
    )
    parser.add_argument("--input_height", default=800, type=int,
                        help="Dummy input height used for FLOPs/MACs measurement")
    parser.add_argument("--input_width", default=1067, type=int,
                        help="Dummy input width used for FLOPs/MACs measurement")
    parser.add_argument("--batch_size_stats", default=1, type=int,
                        help="Dummy batch size used for FLOPs/MACs measurement")
    parser.add_argument("--report_flops", action="store_true",
                        help="Compute FLOPs/MACs with fvcore")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sparse_query_k <= 0 or args.sparse_query_k > args.num_triplets:
        raise ValueError(
            f"sparse_query_k must be in [1, {args.num_triplets}], got {args.sparse_query_k}"
        )

    device = torch.device(args.device)
    model, _, _ = build_model(args)
    model.to(device)
    model.eval()

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report = {
        "num_triplets_configured": args.num_triplets,
        "sparse_query_k": args.sparse_query_k,
        "params_total": params_total,
        "params_trainable": params_trainable,
    }

    if args.report_flops:
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError:
            print(
                "fvcore is required for --report_flops. Install it with: pip install fvcore",
                file=sys.stderr,
            )
            sys.exit(1)

        dummy = torch.randn(
            args.batch_size_stats, 3, args.input_height, args.input_width, device=device
        )

        with torch.no_grad():
            flops = FlopCountAnalysis(model, (dummy,))
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            total_ops = flops.total()
            unsupported_ops = dict(flops.unsupported_ops())

        # fvcore counts one fused multiply-add as one operation, which is closer to MACs.
        report["input_shape"] = [args.batch_size_stats, 3, args.input_height, args.input_width]
        report["macs_fvcore"] = total_ops
        report["flops_mul_add_x2"] = total_ops * 2
        report["unsupported_ops"] = unsupported_ops

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
