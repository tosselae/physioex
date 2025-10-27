#!/usr/bin/env python3
"""
Dataset Compression Script

This script compresses physiological signal datasets by converting them from float32 to bfloat16
format to reduce storage requirements while maintaining reasonable precision for machine learning tasks.

The script processes datasets with standardized preprocessing (normalization) and converts
the data format to save approximately 50% storage space.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from ml_dtypes import bfloat16
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress physiological signal datasets from float32 to bfloat16 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d sleepedf
  %(prog)s --dataset sleepedf --verbose
  %(prog)s -d sleepedf -i /custom/path -o /output/path
  %(prog)s -d sleepedf -p raw xsleepnet -e 0.001 -v
  %(prog)s --dataset sleepedf --dry-run
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="sleepedf",
        help="Dataset name/path relative to the data folder (default: sleepedf)",
    )

    parser.add_argument(
        "--data-folder", "-i", type=str, required=True, help="Source data folder path"
    )

    parser.add_argument(
        "--dest-folder",
        "-o",
        type=str,
        required=True,
        help="Destination folder for compressed data",
    )

    parser.add_argument(
        "--preprocessing",
        "-p",
        nargs="+",
        default=["raw", "xsleepnet"],
        help="List of preprocessing types to process (default: raw xsleepnet)",
    )

    parser.add_argument(
        "--error-threshold",
        "-e",
        type=float,
        default=0.0,
        help="Threshold for approximation error warnings (default: 0.0 - warn on any error)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be processed without actually doing the compression",
    )

    return parser.parse_args()


def validate_paths(data_folder, dest_folder, dataset):
    """Validate that the required paths exist and are accessible."""
    dataset_path = os.path.join(data_folder, dataset)

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    table_path = os.path.join(dataset_path, "table.csv")
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Dataset table file not found: {table_path}")

    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    return True


def compress_dataset(
    dataset,
    data_folder,
    dest_folder,
    preprocessing_types,
    error_threshold=0.0,
    verbose=False,
    dry_run=False,
):
    """
    Compress a dataset from float32 to bfloat16 format.

    Args:
        dataset (str): Dataset name/path relative to data folder
        data_folder (str): Source data folder path
        dest_folder (str): Destination folder for compressed data
        preprocessing_types (list): List of preprocessing types to process
        error_threshold (float): Threshold for approximation error warnings
        verbose (bool): Enable verbose output
        dry_run (bool): Show what would be processed without actually doing it
    """
    dest_dataset_folder = os.path.join(dest_folder, dataset)

    if verbose:
        print(f"Processing dataset: {dataset}")
        print(f"Source folder: {data_folder}")
        print(f"Destination folder: {dest_dataset_folder}")
        print(f"Preprocessing types: {preprocessing_types}")

    if dry_run:
        print(
            f"[DRY RUN] Would process dataset {dataset} with preprocessing: {preprocessing_types}"
        )
        return

    for p in preprocessing_types:
        if verbose:
            print(f"Processing preprocessing type: {p}")

        # create the preprocessing folder
        os.makedirs(os.path.join(dest_dataset_folder, p), exist_ok=True)

        data_path = os.path.join(data_folder, dataset, p)
        table = pd.read_csv(os.path.join(data_folder, dataset, "table.csv"))

        table = table.drop(columns=["Unnamed: 0", "raw", "xsleepnet", "labels"])

        # save the new table in the destination folder
        table.to_csv(os.path.join(dest_dataset_folder, "table.csv"), index=False)

        # the labels folder can be entirely copied as they are already minimal size
        os.system(
            f"cp -r {os.path.join(data_folder, dataset, 'labels')} {dest_dataset_folder}"
        )

        scaling = np.load(os.path.join(data_path, "scaling.npz"))

        # we can copy this file also as it is already minimal size
        os.system(
            f"cp {os.path.join(data_path, 'scaling.npz')} {dest_dataset_folder}/{p}/"
        )

        mean = scaling["mean"]
        std = scaling["std"]

        input_shape = mean.shape

        # iterate over the table rows
        for i, row in tqdm(
            table.iterrows(),
            total=table.shape[0],
            desc=f"Processing {dataset} {p} data",
        ):
            subject_id = row["subject_id"]
            num_windows = row["num_windows"]

            subject_shape = (num_windows,) + input_shape

            data_path = os.path.join(data_folder, dataset, p, f"{subject_id}.npy")
            subject_signal = np.memmap(
                data_path, dtype="float32", mode="r", shape=subject_shape
            )

            # read the subject data
            subject_signal = subject_signal[:]

            # normalize the subject data
            subject_signal = (subject_signal - mean) / std

            original_signal = torch.from_numpy(subject_signal).to(torch.bfloat16)
            subject_signal = subject_signal.astype(bfloat16)

            mm = np.memmap(
                os.path.join(dest_dataset_folder, p, f"{subject_id}.npy"),
                dtype=bfloat16,
                mode="w+",
                shape=subject_signal.shape,
            )

            mm[:] = subject_signal
            mm.flush()

            subject_signal = torch.from_numpy(mm.astype(np.float32)).float()
            original_signal = original_signal.float()

            error = torch.abs(original_signal - subject_signal).mean().item()

            if error > error_threshold:
                print(
                    f"Warning: High approximation error for {subject_id}: {error:.6f}"
                )

            if verbose and error > 0:
                print(f"Approximation error for {subject_id}: {error:.6f}")


def main():
    """Main function to run the dataset compression script."""
    try:
        args = parse_arguments()

        data_folder = args.data_folder
        dest_folder = args.dest_folder

        print("Starting dataset compression...")
        print("Source data folder:", data_folder)
        print("Destination folder:", dest_folder)

        if args.verbose:
            print("Arguments:")
            print(f"  Dataset: {args.dataset}")
            print(f"  Data folder: {data_folder}")
            print(f"  Destination folder: {dest_folder}")
            print(f"  Preprocessing types: {args.preprocessing}")
            print(f"  Error threshold: {args.error_threshold}")
            print(f"  Dry run: {args.dry_run}")
            print()

        # Validate paths
        if not args.dry_run:
            validate_paths(data_folder, dest_folder, args.dataset)

        # Compress the dataset
        compress_dataset(
            dataset=args.dataset,
            data_folder=data_folder,
            dest_folder=dest_folder,
            preprocessing_types=args.preprocessing,
            error_threshold=args.error_threshold,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            print("Dataset compression completed successfully!")
            print(
                f"Compressed data saved to: {os.path.join(dest_folder, args.dataset)}"
            )

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
