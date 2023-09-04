import mmcv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("cfg_path", type=str, help="Path to the cfg file")
args = parser.parse_args()

cfg = mmcv.Config.fromfile(args.cfg_path)
print(cfg.pretty_text)
