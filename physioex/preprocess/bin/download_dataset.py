#!/usr/bin/env python3
"""
Dataset Download Script

This script downloads compressed physiological signal datasets from Hugging Face Hub.
The datasets are in bfloat16 format to minimize storage and transfer requirements
while maintaining reasonable precision for machine learning tasks.

The script downloads pre-compressed datasets that were processed using the compression
script and stored on Hugging Face Hub with anonymized repository names for privacy.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError


# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def disable(cls):
        """Disable colors for non-supporting terminals."""
        cls.HEADER = ""
        cls.OKBLUE = ""
        cls.OKCYAN = ""
        cls.OKGREEN = ""
        cls.WARNING = ""
        cls.FAIL = ""
        cls.ENDC = ""
        cls.BOLD = ""
        cls.UNDERLINE = ""


# Disable colors if not in a TTY (e.g., when piping output)
if not sys.stdout.isatty():
    Colors.disable()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download compressed physiological signal datasets from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d sleepedf
  %(prog)s --dataset hmc --verbose
  %(prog)s -d hmc -p raw xsleepnet -o /custom/output/path
  %(prog)s -d dcsm -p xsleepnet --dest-folder /data/downloads -v
  %(prog)s --dataset dcsm --dry-run
  %(prog)s --list-datasets

Note: Some datasets require special access permissions. Contact the maintainer 
      if you need access to restricted datasets.
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="sleepedf",
        help="Dataset name to download (default: sleepedf). Use --list-datasets to see available options",
    )

    parser.add_argument(
        "--dest-folder",
        "-o",
        type=str,
        default=None,
        help="Destination folder for downloaded data. If not specified, uses $VSC_SCRATCH_NODE/sleep-data/",
    )

    parser.add_argument(
        "--preprocessing",
        "-p",
        nargs="+",
        default=["xsleepnet"],
        help="List of preprocessing types to download (default: xsleepnet). Options: raw, xsleepnet",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )

    parser.add_argument(
        "--list-datasets",
        "-l",
        action="store_true",
        help="List all available datasets and exit",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force download even if files already exist locally",
    )

    return parser.parse_args()


def list_available_datasets():
    """List all available datasets with modern formatting and colors."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}🔬 PhysioEx Dataset Repository{Colors.ENDC}")
    print(
        f"{Colors.BOLD}═══════════════════════════════════════════════════════════════════════════════{Colors.ENDC}\n"
    )

    # Public Datasets Section
    print(f"{Colors.BOLD}{Colors.OKGREEN}📂 Public Datasets{Colors.ENDC}")
    print(
        f"{Colors.OKGREEN}├─ Available immediately, no special permissions required{Colors.ENDC}"
    )
    print(f"{Colors.OKGREEN}└─ Simply use the dataset name to download{Colors.ENDC}\n")

    public_datasets = [
        (
            "sleepedf",
            "SleepEDF Dataset",
            "https://physionet.org/content/sleep-edfx/1.0.0/",
        ),
        (
            "hmc",
            "HMC Sleep Staging Dataset",
            "https://physionet.org/content/hmc-sleep-staging/1.1/",
        ),
        (
            "dcsm",
            "DCSM Dataset",
            "https://erda.ku.dk/public/archives/db553715ecbe1f3ac66c1dc569826eef/published-archive.html",
        ),
    ]

    for name, title, url in public_datasets:
        print(f"   {Colors.OKCYAN}▶ {title}{Colors.ENDC}")
        print(f"     {Colors.BOLD}Name:{Colors.ENDC} '{name}'")
        print(f"     {Colors.BOLD}URL:{Colors.ENDC} {url}")
        print()

    # NSRR Datasets Section
    print(f"{Colors.BOLD}{Colors.WARNING}🔐 NSRR Archive Datasets{Colors.ENDC}")
    print(
        f"{Colors.WARNING}├─ Requires access from NSRR archive: https://sleepdata.org/{Colors.ENDC}"
    )
    print(
        f"{Colors.WARNING}├─ Get NSRR access first, then contact: guido.gagliardi@phd.unipi.it{Colors.ENDC}"
    )
    print(
        f"{Colors.WARNING}└─ Access credentials will be provided after verification{Colors.ENDC}\n"
    )

    nsrr_datasets = ["SHHS", "MESA", "MrOS", "HPAP", "WSC"]
    for i, dataset in enumerate(nsrr_datasets):
        prefix = "├─" if i < len(nsrr_datasets) - 1 else "└─"
        print(f"   {Colors.WARNING}{prefix} {dataset} Dataset{Colors.ENDC}")
    print()

    # MASS Dataset Section
    print(f"{Colors.BOLD}{Colors.FAIL}🏥 MASS Dataset{Colors.ENDC}")
    print(f"{Colors.FAIL}├─ Montreal Archive of Sleep Studies{Colors.ENDC}")
    print(f"{Colors.FAIL}├─ Get access from: https://ceams-carsm.ca/mass/{Colors.ENDC}")
    print(f"{Colors.FAIL}└─ Then contact: guido.gagliardi@phd.unipi.it{Colors.ENDC}\n")

    # Hospital Datasets Section
    print(f"{Colors.BOLD}{Colors.HEADER}🏥 UZ Leuven Hospital Datasets{Colors.ENDC}")
    print(f"{Colors.HEADER}├─ Restricted to affiliated researchers only{Colors.ENDC}")
    print(
        f"{Colors.HEADER}└─ Contact: guido.gagliardi@phd.unipi.it for details{Colors.ENDC}\n"
    )

    # Coming Soon Section
    print(f"{Colors.BOLD}{Colors.OKCYAN}🐭 Coming Soon{Colors.ENDC}")
    print(f"{Colors.OKCYAN}└─ Mice Sleep Datasets - Stay tuned!{Colors.ENDC}\n")

    # Instructions
    print(f"{Colors.BOLD}💡 How to Use:{Colors.ENDC}")
    print(
        f"   {Colors.OKBLUE}• Public datasets:{Colors.ENDC} python download_dataset.py -d <name>"
    )
    print(
        f"   {Colors.OKBLUE}• Restricted datasets:{Colors.ENDC} Contact maintainer for access credentials"
    )
    print(f"   {Colors.OKBLUE}• Help:{Colors.ENDC} python download_dataset.py --help")

    print(
        f"\n{Colors.BOLD}═══════════════════════════════════════════════════════════════════════════════{Colors.ENDC}"
    )
    print(
        f"{Colors.BOLD}Total: {len(public_datasets)} public datasets + multiple restricted datasets available{Colors.ENDC}"
    )
    print(
        f"{Colors.BOLD}═══════════════════════════════════════════════════════════════════════════════{Colors.ENDC}\n"
    )


def download_dataset(
    dataset_name,
    dest_folder,
    preprocessing_types,
    verbose=False,
    dry_run=False,
    force=False,
):
    """
    Download a compressed dataset from Hugging Face Hub.

    Args:
        dataset_name (str): Name of the dataset to download
        dest_folder (str): Destination folder for downloaded data
        preprocessing_types (list): List of preprocessing types to download
        verbose (bool): Enable verbose output
        dry_run (bool): Show what would be downloaded without actually downloading
        force (bool): Force download even if files already exist locally
    """
    dataset_id = dataset_name
    local_path = os.path.join(dest_folder, dataset_name)

    if verbose:
        print(f"Downloading dataset: {dataset_name}")
        print(f"Hugging Face repo: 4rooms/{dataset_id}")
        print(f"Destination folder: {local_path}")
        print(f"Preprocessing types: {preprocessing_types}")

    if dry_run:
        print(
            f"[DRY RUN] Would download dataset '{dataset_name}' with preprocessing: {preprocessing_types}"
        )
        print(f"[DRY RUN] Destination: {local_path}")
        return True

    # Check if files already exist
    if os.path.exists(local_path) and not force:
        existing_files = os.listdir(local_path)
        if existing_files:
            if verbose:
                print(f"Files already exist in {local_path}. Use --force to overwrite.")
            response = input(
                f"Dataset folder {local_path} already exists. Continue? [y/N]: "
            )
            if response.lower() not in ["y", "yes"]:
                print("Download cancelled.")
                return False

    # Create destination directory
    os.makedirs(local_path, exist_ok=True)

    # Build allow patterns for the specific preprocessing types
    allow_patterns = ["table.csv", "labels/*"]
    for prep_type in preprocessing_types:
        allow_patterns.append(f"{prep_type}/*")

    if verbose:
        print(f"Download patterns: {allow_patterns}")

    try:
        print(
            f"\n{Colors.OKCYAN}{Colors.BOLD}📥 Downloading dataset '{dataset_name}' from Hugging Face Hub...{Colors.ENDC}"
        )
        snapshot_download(
            repo_id=f"4rooms/{dataset_id}",
            repo_type="dataset",
            local_dir=local_path,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )

        print(
            f"{Colors.OKGREEN}{Colors.BOLD}✅ Download completed successfully!{Colors.ENDC}"
        )
        return True

    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            print(
                f"\n{Colors.FAIL}{Colors.BOLD}❌ Error: Repository not found{Colors.ENDC}"
            )
            print(
                f"{Colors.FAIL}   Repository 4rooms/{dataset_id} is not accessible.{Colors.ENDC}"
            )
            print(
                f"{Colors.WARNING}   This dataset may require special access permissions.{Colors.ENDC}"
            )
            print(
                f"{Colors.OKCYAN}   📧 Contact: guido.gagliardi@phd.unipi.it for access credentials{Colors.ENDC}"
            )
        elif e.response.status_code == 401:
            print(
                f"\n{Colors.FAIL}{Colors.BOLD}🔐 Error: Authentication failed{Colors.ENDC}"
            )
            print(
                f"{Colors.FAIL}   This dataset requires special access permissions.{Colors.ENDC}"
            )
            print(
                f"{Colors.OKCYAN}   📧 Contact: guido.gagliardi@phd.unipi.it for access credentials{Colors.ENDC}"
            )
            print(
                f"{Colors.WARNING}   You may also need to login with: 'huggingface-cli login'{Colors.ENDC}"
            )
        else:
            print(f"\n{Colors.FAIL}{Colors.BOLD}⚠️  HTTP Error: {e}{Colors.ENDC}")
        return False
    except Exception as e:
        print(
            f"\n{Colors.FAIL}{Colors.BOLD}💥 Unexpected error during download: {e}{Colors.ENDC}"
        )
        return False


def main():
    """Main function to run the dataset download script."""
    try:
        args = parse_arguments()

        # Handle list datasets option
        if args.list_datasets:
            list_available_datasets()
            sys.exit(0)

        dest_folder = args.dest_folder

        if args.verbose:
            print("Arguments:")
            print(f"  Dataset: {args.dataset}")
            print(f"  Destination folder: {dest_folder}")
            print(f"  Preprocessing types: {args.preprocessing}")
            print(f"  Force download: {args.force}")
            print(f"  Dry run: {args.dry_run}")
            print()

        # Download the dataset
        success = download_dataset(
            dataset_name=args.dataset,
            dest_folder=dest_folder,
            preprocessing_types=args.preprocessing,
            verbose=args.verbose,
            dry_run=args.dry_run,
            force=args.force,
        )

        if success and not args.dry_run:
            print(
                f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 Dataset download completed successfully!{Colors.ENDC}"
            )
            print(
                f"{Colors.OKCYAN}📂 Data saved to: {os.path.join(dest_folder, args.dataset)}{Colors.ENDC}"
            )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(
            f"\n{Colors.WARNING}{Colors.BOLD}⏹️  Operation cancelled by user.{Colors.ENDC}"
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}{Colors.BOLD}❌ Error: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
