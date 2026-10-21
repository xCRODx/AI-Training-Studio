import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compile_runtime.converter import convert_checkpoint_to_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert base_finetuning to RetBit GGUF v3")
    parser.add_argument("--source", default="checkpoints_finetuning/base_finetuning")
    parser.add_argument("--output", default="artifacts/bitnet/base_finetuning-retbit.gguf")
    args = parser.parse_args()
    result = convert_checkpoint_to_gguf(args.source, args.output, print)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
