#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    quick_mode = "--quick" in sys.argv
    decompress_mode = "--decompress" in sys.argv

    if decompress_mode:
        from src.decompress import run_decompression_test
        results = run_decompression_test()
        if results:
            rt = results.get("round_trips", [])
            print(f"\nDecompression test complete. {len(rt)} round-trips recorded.")
        return

    if quick_mode:
        from src import config
        config.BUDGET_LEVELS = [1000, 250]
        print("[QUICK MODE: 2 budget levels only]\n")

    from src.orchestrator import run_experiment
    results = run_experiment()

    if results:
        print(f"\nExperiment complete. {results['total_runs']} runs recorded.")
        print(f"Check results/ for data and plots.")

if __name__ == "__main__":
    main()
