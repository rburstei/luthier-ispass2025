import math
from common import read_yaml_cfg
import argparse
import os
import pickle
import csv


def parse_and_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Instruction count runner script")
    parser.add_argument("--hecbench_dir", type=str,
                        default="/work/HeCBench",
                        help="location of the HeCBench repo")
    parser.add_argument("--specs_yaml", type=str,
                        default="scripts/specs.yaml",
                        help="supply your own specs yaml file")
    parser.add_argument("--out_csv", type=str,
                        default="amd-full-instrcount-results.csv",
                        help="output CSV file path")

    args = parser.parse_args()
    return args


RESULT_PKLE_FILE_NAME = "amd-full-instrcount-results.pkl"


CSV_KEYS = [
    "Number of Instructions",
    "Un-instrumented SYCL Kernel Runtime (us)",
    "Instrumented SYCL Kernel Runtime (us)",
    "Instrumented Luthier Kernel Runtime (us)",
    "Un-instrumented with Luthier-freamwork, SYCL Kernel Runtime (us)",
    "Un-instrumented with Luthier-freamwork, Luthier Kernel Runtime (us)",
]


def main():
    args = parse_and_validate_args()
    benchmark_cfg = read_yaml_cfg(args.specs_yaml)

    benchmarks = benchmark_cfg["InstrCount"]["benchmarks"]
    programming_models = benchmark_cfg["InstrCount"]["programming_models"]

    # Collect rows for CSV
    rows = []

    for bench in benchmarks:
        for programming_model in programming_models:
            benchmark_folder = os.path.join(
                args.hecbench_dir, "src", f"{bench}-{programming_model}"
            )
            pkl_path = os.path.join(benchmark_folder, RESULT_PKLE_FILE_NAME)

            if not os.path.isfile(pkl_path):
                raise FileNotFoundError(
                    f"Missing results pickle: {pkl_path}"
                )

            with open(pkl_path, "rb") as f:
                data_point = pickle.load(f)

            row = {
                "benchmark": bench,
                "programming_model": programming_model,
            }

            # Extract requested keys (leave blank if missing)
            for k in CSV_KEYS:
                row[k] = data_point.get(k, "")

            rows.append(row)

    # Write CSV
    fieldnames = ["benchmark", "programming_model"] + CSV_KEYS
    with open(args.out_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV with {len(rows)} rows to: {args.out_csv}")


if __name__ == "__main__":
    main()
