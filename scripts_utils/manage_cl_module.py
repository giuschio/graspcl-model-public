"""Manage the portable files and online data of an existing CL module."""

import argparse

from grasp_learning_cl.cl_module_probabilistic import ContinualLearningModule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cl_module_path", required=True, help="Existing CL-module directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "internalize-encoder",
        help="Copy the configured encoder checkpoint into the CL-module directory",
    )
    subparsers.add_parser(
        "drop-online-data",
        help="Delete online scoring and recall data after confirmation",
    )
    args = parser.parse_args()

    cl_module = ContinualLearningModule.load_inplace(args.cl_module_path, device="cpu")
    if args.command == "internalize-encoder":
        encoder_path = cl_module.internalize_encoder()
        print(f"internal_encoder={encoder_path}")
    elif args.command == "drop-online-data":
        cl_module.drop_online_data()


if __name__ == "__main__":
    main()
