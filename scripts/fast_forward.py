import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.advance_day import main as advance_one_day
from scripts.replay_batch import main as replay_one_day
from src.label_injector import main as release_labels


def main():
    parser = argparse.ArgumentParser(
        description="Advance the simulated clock N days, replaying traffic and releasing labels at each step."
    )
    parser.add_argument("days", type=int, help="number of simulated days to fast-forward")
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="seconds to pause between days, useful if API_URL points at a free-tier host you don't want to hammer",
    )
    args = parser.parse_args()

    for i in range(args.days):
        print(f"--- day {i + 1}/{args.days} ---")
        advance_one_day()
        replay_one_day()
        release_labels()
        if args.sleep:
            time.sleep(args.sleep)

    print(f"fast-forwarded {args.days} simulated days")


if __name__ == "__main__":
    main()