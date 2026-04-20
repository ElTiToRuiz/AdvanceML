"""
Project entry point — Predictive Analysis & Explainability in Financial Markets.

Dispatches to the requested activity:
    python main.py 1     # runs src/activity1 pipeline (EDA)
    python main.py 2     # runs src/activity2 pipeline
    python main.py       # lists available activities

Each activity lives in its own package under src/ with its own main.py
and a utils/ subfolder for deeper modules.
"""

import argparse
import importlib
import sys

ACTIVITIES = {
    "1": "src.activity1.main",
    "2": "src.activity2.main",
}


def run(activity: str) -> None:
    if activity not in ACTIVITIES:
        print(f"Activity '{activity}' not found. Options: {', '.join(ACTIVITIES)}")
        sys.exit(1)
    module = importlib.import_module(ACTIVITIES[activity])
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a project activity.")
    parser.add_argument(
        "activity",
        nargs="?",
        choices=list(ACTIVITIES),
        help="Activity number to run (1 or 2).",
    )
    args = parser.parse_args()

    if args.activity is None:
        print("Available activities:")
        for key, mod in ACTIVITIES.items():
            print(f"  {key} → {mod}")
        print("\nUsage: python main.py <number>")
        return

    run(args.activity)


if __name__ == "__main__":
    main()
