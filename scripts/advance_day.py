import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from serving.db import advance_batch, get_engine


def main():
    engine = get_engine()
    new_batch = advance_batch(engine)
    print(f"advanced to simulated batch {new_batch}")


if __name__ == "__main__":
    main()