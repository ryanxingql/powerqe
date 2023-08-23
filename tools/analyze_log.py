"""Copyright 2023 RyanXingQL.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import json
import os.path as osp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np


def read_json(json_path, losses, metrics):
    """
    Examples:
        {
            "exp_name": "exp2.6",
            "mmedit Version": "0.16.1",
            "seed": 0,
            "env_info": "test"
        }
        {
            "mode": "train",
            "epoch": 1,
            "iter": 100,
            "lr": {
                "generator": 5e-05
            },
            "memory": 324,
            "data_time": 0.00106,
            "loss_pix": 0.01336,
            "loss": 0.01336,
            "time": 0.15789
        }
        {
            "mode": "val",
            "epoch": 1,
            "iter": 50000,
            "lr": {
                "generator": 5e-05
            },
            "PSNR": 34.24646,
            "SSIM": 0.95762
        }
        {
            "mode": "val",
            "epoch": 1,
            "iter": 50000,
            "lr": {
                "generator": 5e-05
            }
        }
    """

    with open(json_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if "mode" in data and data["mode"] == "train":
                iters = data["iter"]
                loss = data["loss"]
                losses[iters] = loss
            if "mode" in data and data["mode"] == "val":
                for metric in metrics.keys():
                    if metric in data:
                        iters = data["iter"]
                        result = data[metric]
                        metrics[metric][iters] = result

    return losses, metrics


def plot_curve(data, ylabel, smooth=False, save_path=""):
    keys = list(data.keys())
    values = list(data.values())

    # Moving average
    if smooth:
        window_size = 9
        assert window_size % 2 == 1
        keys = keys[window_size // 2 : -(window_size // 2)]
        values = np.convolve(values, np.ones(window_size) / window_size, mode="valid")

    plt.plot(keys, values)
    plt.xlabel("Iters")
    plt.ylabel(ylabel)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Parse JSON file")
    parser.add_argument("json_path", type=str, help="Path to the JSON file")
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Path to save the PNG"
    )

    args = parser.parse_args()

    if osp.isdir(args.json_path):
        json_files = glob(osp.join(args.json_path, "*.json"))
        sorted_json_files = sorted(json_files, key=lambda x: osp.getmtime(x))
        print(f"{len(sorted_json_files)} json files are found.")
    else:
        sorted_json_files = [args.json_path]

    if not args.save_dir:
        args.save_dir = osp.dirname(sorted_json_files[0])

    # Read JSON
    losses = dict()
    metrics = dict(PSNR=dict(), SSIM=dict())
    for json_file in sorted_json_files:
        losses, metrics = read_json(json_file, losses=losses, metrics=metrics)

    # Plot
    save_path = osp.join(args.save_dir, "losses.png")
    plot_curve(data=losses, ylabel="Loss", smooth=True, save_path=save_path)

    for k, v in metrics.items():
        if v:
            save_path = osp.join(args.save_dir, f"{k}.png")
            plot_curve(data=v, ylabel=k, save_path=save_path)


if __name__ == "__main__":
    main()
