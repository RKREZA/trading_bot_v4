import argparse
import sys
print(f"LITE_ARGS: sys.argv={sys.argv}")
parser = argparse.ArgumentParser()
parser.add_argument("--symbol", type=str)
args = parser.parse_args()
print(f"LITE_ARGS: SUCCESS. args.symbol={args.symbol}")
