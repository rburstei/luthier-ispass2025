#!/usr/bin/env python3

import os
import argparse
from typing import Tuple
import pickle
import re

from common import read_yaml_cfg, capture_subprocess_output


RESULT_PKLE_FILE_NAME = "amd-full2-instrcount-results.pkl"


def parse_and_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Instruction count runner script")
    parser.add_argument(
        "--hecbench_dir",
        type=str,
        default="/work/HeCBench",
        help="location of the HeCBench repo",
    )
    parser.add_argument(
        "--luthier_instr_count_tool_path",
        type=str,
        default="/work/Luthier/build/examples/InstrCount/libLuthierInstrCount.so",
        help="location of the luthier instruction count tool",
    )
    parser.add_argument(
        "--dump_stdout_stderr",
        action="store_true",
        help="dump the stdout and stderr of each experiment to screen",
    )
    parser.add_argument(
        "--overwrite_results",
        action="store_true",
        help="overwrite results of benchmarks from a previous run",
    )
    parser.add_argument(
        "--specs_yaml",
        type=str,
        default="scripts/specs.yaml",
        help="location of specs file",
    )

    return parser.parse_args()


def luthier_get_instr_count_tool_results(
    stdout: str,
    stderr: str,
    tool_run: bool = True,
    instrumented: bool = True,
) -> Tuple[int, int, int]:
    combined = stdout + "\n" + stderr
    lines = combined.splitlines()

    instruction_count = -1
    sycl_kernel_runtime = -1     # us (converted from ms)
    luthier_kernel_time = -1     # us (already us)

    sycl_ms_re = re.compile(
        r"SYCL_MEASUREMENT:\s*Total kernel execution time on GPU:\s*([0-9]*\.?[0-9]+)\s*\(ms\)"
    )

    for line in reversed(lines):
        if not tool_run and not instrumented:
            if sycl_kernel_runtime != -1:
                break
        elif tool_run and not instrumented:
            if sycl_kernel_runtime != -1 and luthier_kernel_time != -1:
                break
        elif tool_run and instrumented:
            if (
                instruction_count != -1
                and sycl_kernel_runtime != -1
                and luthier_kernel_time != -1
            ):
                break

        if "Total number of counted instructions:" in line:
            instruction_count = int(
                line.replace("Total number of counted instructions:", "")
                    .replace(".", "")
                    .strip()
            )
            continue

        if "Total kernel run time (us):" in line:
            luthier_kernel_time = int(
                line.replace("Total kernel run time (us):", "")
                    .replace(".", "")
                    .strip()
            )
            continue

        m = sycl_ms_re.search(line)
        if m:
            sycl_kernel_runtime = int(round(float(m.group(1)) * 1000.0))
            continue

    if sycl_kernel_runtime == -1:
        raise EOFError("Failed to find the kernel runtime (SYCL_MEASUREMENT line).")

    if tool_run and luthier_kernel_time == -1:
        raise EOFError(
            "Failed to find the Luthier kernel time (Total kernel run time (us) line)."
        )

    if tool_run and instrumented and instruction_count == -1:
        raise EOFError("Failed to find the instruction count.")

    return instruction_count, sycl_kernel_runtime, luthier_kernel_time


def main():
    args = parse_and_validate_args()
    benchmark_cfg = read_yaml_cfg(args.specs_yaml)

    for bench in benchmark_cfg["InstrCount"]["benchmarks"]:
        for programming_model in benchmark_cfg["InstrCount"]["programming_models"]:
            out = {}

            cfgs = benchmark_cfg["HeCBench"][bench]["programming_models"][programming_model]
            run_flags = eval(cfgs["run_command"])  # assumed trusted YAML

            benchmark_folder = os.path.join(
                args.hecbench_dir, "src", f"{bench}-{programming_model}"
            )

            result_path = os.path.join(benchmark_folder, RESULT_PKLE_FILE_NAME)
            if os.path.exists(result_path) and not args.overwrite_results:
                print(f"Skipping {bench}-{programming_model} (results already exist).")
                continue

            # ------------------------------------------------------------------
            # Un-instrumented run
            # ------------------------------------------------------------------
            env_vars = os.environ.copy()
            print(
                f"Running un-instrumented {bench}-{programming_model}: {' '.join(run_flags)}"
            )

            rc, stdout, stderr = capture_subprocess_output(
                args=run_flags,
                cwd=benchmark_folder,
                env=env_vars,
                dump_stdout_stderr=args.dump_stdout_stderr,
            )
            if rc:
                raise ChildProcessError("Un-instrumented run failed.")

            _, sycl_runtime, _ = luthier_get_instr_count_tool_results(
                stdout, stderr, tool_run=False, instrumented=False
            )
            out["Un-instrumented SYCL Kernel Runtime (us)"] = sycl_runtime

            # ------------------------------------------------------------------
            # Instrumented run
            # ------------------------------------------------------------------
            env_vars["LD_PRELOAD"] = args.luthier_instr_count_tool_path
            env_vars["HIP_ENABLE_DEFERRED_LOADING"] = "0"

            print(
                f"Running instrumented {bench}-{programming_model}: {' '.join(run_flags)}"
            )

            rc, stdout, stderr = capture_subprocess_output(
                args=run_flags,
                cwd=benchmark_folder,
                env=env_vars,
                dump_stdout_stderr=args.dump_stdout_stderr,
            )
            if rc:
                raise ChildProcessError("Instrumented run failed.")

            instr_cnt, sycl_runtime, luthier_runtime = (
                luthier_get_instr_count_tool_results(stdout, stderr, True, True)
            )

            out["Number of Instructions"] = instr_cnt
            out["Instrumented SYCL Kernel Runtime (us)"] = sycl_runtime
            out["Instrumented Luthier Kernel Runtime (us)"] = luthier_runtime

            # ------------------------------------------------------------------
            # Un-instrumented with Luthier framework
            # ------------------------------------------------------------------
            env_vars["LUTHIER_ARGS"] = "--instr-end-interval=0"

            print(
                f"Running un-instrumented with Luthier framework "
                f"{bench}-{programming_model}"
            )

            rc, stdout, stderr = capture_subprocess_output(
                args=run_flags,
                cwd=benchmark_folder,
                env=env_vars,
                dump_stdout_stderr=args.dump_stdout_stderr,
            )
            if rc:
                raise ChildProcessError(
                    "Un-instrumented with Luthier framework run failed."
                )

            _, sycl_runtime, luthier_runtime = (
                luthier_get_instr_count_tool_results(stdout, stderr, True, False)
            )

            out[
                "Un-instrumented with Luthier-framework, SYCL Kernel Runtime (us)"
            ] = sycl_runtime
            out[
                "Un-instrumented with Luthier-framework, Luthier Kernel Runtime (us)"
            ] = luthier_runtime

            print(out)

            with open(result_path, "wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Ran {bench} benchmarks.")


if __name__ == "__main__":
    main()
