"""STAGE 9: Model Evaluation Script.
Executes the CDM Benchmark Suite against the test dataset and outputs benchmark results.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.benchmark_suite import CDMBenchmarkSuite


def main():
    suite = CDMBenchmarkSuite()
    report = suite.run_full_suite()
    print("Stage 9 evaluation completed successfully.")


if __name__ == "__main__":
    main()
