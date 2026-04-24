"""
SDC2026 KAU AE Team — Pipeline Runner

Runs the full CDM analysis pipeline in order. Stops immediately if any step fails.
Supports resuming from a specific step to avoid re-running expensive steps.

Usage:
    # Full pipeline from scratch:
    python Scripts/run_pipeline.py --kvn-dir path/to/kvn/files

    # Resume from step 3 (step1 + step2 already done):
    python Scripts/run_pipeline.py --from-step 3

    # Run only evaluation and inference (steps 3b, 4):
    python Scripts/run_pipeline.py --from-step 3b --to-step 4

    # Dry run — print what would run without executing:
    python Scripts/run_pipeline.py --dry-run

Steps:
    1   → step1_parse_kvn.py         (KVN files → CSV)
    2   → step2_prepare_sequences.py  (CSV → sequences)
    3   → step3_train_model.py        (train BiGRU)
    3b  → step3b_evaluate_proxy_confidence.py  (offline evaluation gate)
    4   → step4_inference_dashboard.py          (production inference)
    5   → step5_visualize.py          (figures)
    5b  → step5b_detailed_reports.py  (detailed reports)
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


# ---------------------------------------------------------------------------
# Step definitions (order matters)
# ---------------------------------------------------------------------------

STEPS = [
    {
        'id': '1',
        'name': 'KVN Parser',
        'script': 'step1_parse_kvn.py',
        'description': 'Parse KVN files → parsed_cdm_data.csv',
        'requires_kvn_dir': True,
    },
    {
        'id': '2',
        'name': 'Sequence Preparation',
        'script': 'step2_prepare_sequences.py',
        'description': 'CSV → numpy sequences (X/Y train/val/test)',
        'requires_kvn_dir': False,
    },
    {
        'id': '3',
        'name': 'Model Training',
        'script': 'step3_train_model.py',
        'description': 'Train BiGRU self-supervised model',
        'requires_kvn_dir': False,
    },
    {
        'id': '3b',
        'name': 'Offline Evaluation Gate',
        'script': 'step3b_evaluate_proxy_confidence.py',
        'description': 'Validate model quality before production inference',
        'requires_kvn_dir': False,
        'gate': True,  # If this fails, stop — do not run step4
    },
    {
        'id': '4',
        'name': 'Inference Dashboard',
        'script': 'step4_inference_dashboard.py',
        'description': 'Production inference → threat/confidence scores',
        'requires_kvn_dir': False,
    },
    {
        'id': '5',
        'name': 'Visualization',
        'script': 'step5_visualize.py',
        'description': 'Generate performance figures',
        'requires_kvn_dir': False,
    },
    {
        'id': '5b',
        'name': 'Detailed Reports',
        'script': 'step5b_detailed_reports.py',
        'description': 'Generate per-event detailed reports',
        'requires_kvn_dir': False,
    },
]

STEP_ORDER = ['1', '2', '3', '3b', '4', '5', '5b']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def step_index(step_id):
    try:
        return STEP_ORDER.index(step_id)
    except ValueError:
        return -1


def get_step(step_id):
    for s in STEPS:
        if s['id'] == step_id:
            return s
    return None


def print_banner():
    print("=" * 70)
    print("  SDC2026 KAU AE Team — Pipeline Runner")
    print("  DebriSolver Space Data Challenge 2026")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_step_header(step, step_num, total):
    print()
    print(f"{'─' * 70}")
    print(f"  [{step_num}/{total}] Step {step['id']}: {step['name']}")
    print(f"  {step['description']}")
    print(f"{'─' * 70}")


def print_summary(results):
    print()
    print("=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    for step_id, status, duration in results:
        step = get_step(step_id)
        icon = '✓' if status == 'PASSED' else ('↷' if status == 'SKIPPED' else '✗')
        dur = f"({duration:.1f}s)" if duration else ""
        print(f"  {icon} Step {step_id}: {step['name']:30s} {status:8s} {dur}")
    print()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def run_step(step, scripts_dir, kvn_dir=None, dry_run=False):
    """Run a single step. Returns (success, duration_seconds)."""
    script_path = os.path.join(scripts_dir, step['script'])

    if not os.path.exists(script_path):
        print(f"  ✗ Script not found: {script_path}")
        return False, 0

    cmd = [sys.executable, script_path]

    if step.get('requires_kvn_dir') and kvn_dir:
        cmd += ['--kvn-dir', kvn_dir]

    print(f"  $ {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN — not executed]")
        return True, 0

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=scripts_dir,
            check=False,  # We handle non-zero return ourselves
        )
        duration = time.time() - start

        if result.returncode != 0:
            print(f"\n  ✗ Step {step['id']} FAILED (exit code {result.returncode})")
            return False, duration

        print(f"\n  ✓ Step {step['id']} completed in {duration:.1f}s")
        return True, duration

    except KeyboardInterrupt:
        print(f"\n  ✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        duration = time.time() - start
        print(f"\n  ✗ Step {step['id']} raised an unexpected error: {e}")
        return False, duration


def main():
    parser = argparse.ArgumentParser(
        description='Run the SDC2026 KAU AE Team pipeline end-to-end',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--kvn-dir', default='.',
        help='Path to KVN files directory (required for step 1, default: current dir)'
    )
    parser.add_argument(
        '--from-step', default='1', choices=STEP_ORDER,
        help='Start from this step (skip earlier steps). Default: 1'
    )
    parser.add_argument(
        '--to-step', default='5b', choices=STEP_ORDER,
        help='Stop after this step. Default: 5b (run all)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would run without actually executing'
    )
    args = parser.parse_args()

    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    print_banner()

    # Determine which steps to run
    from_idx = step_index(args.from_step)
    to_idx = step_index(args.to_step)

    if from_idx > to_idx:
        print(f"  ✗ --from-step ({args.from_step}) is after --to-step ({args.to_step})")
        sys.exit(1)

    steps_to_run = [s for s in STEPS if from_idx <= step_index(s['id']) <= to_idx]
    total = len(steps_to_run)

    print(f"  Running {total} step(s): {', '.join(s['id'] for s in steps_to_run)}")
    if args.dry_run:
        print("  [DRY RUN MODE — no scripts will be executed]")

    results = []

    for i, step in enumerate(steps_to_run, 1):
        print_step_header(step, i, total)

        success, duration = run_step(
            step,
            scripts_dir=scripts_dir,
            kvn_dir=args.kvn_dir,
            dry_run=args.dry_run,
        )

        status = 'PASSED' if success else 'FAILED'
        results.append((step['id'], status, duration))

        if not success:
            # Gate steps stop the pipeline on failure
            print()
            if step.get('gate'):
                print(f"  ✗ GATE FAILED: Step {step['id']} ({step['name']}) did not pass.")
                print(f"    The model does not meet quality requirements.")
                print(f"    Step 4 (inference) will NOT run until this is resolved.")
            else:
                print(f"  ✗ Step {step['id']} failed. Pipeline halted.")
            print_summary(results)
            sys.exit(1)

    print_summary(results)
    print(f"  ✓ Pipeline completed successfully at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
