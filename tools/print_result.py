import os
import argparse
import glob
import json
import numpy as np

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="")
    return parser

if __name__ == "__main__":
    args = init_parser().parse_args()
    metric_files = sorted(
        glob.glob(os.path.join(args.result_path, "*/metrics.json"))
    )
    with open(metric_files[0], "r") as f:
        example_metric = json.load(f)
    keys = example_metric.keys()
    summary_metrics = {k: [] for k in keys}
    for file in metric_files:
        with open(file, "r") as f:
            tmp_metric = json.load(f)
            for k in keys:
                summary_metrics[k].append(tmp_metric[k])
    print("total test sample number:{}".format(len(summary_metrics["acc"])))
    for k in keys:
        summary_metrics[k] = np.mean(np.array(summary_metrics[k]))
    for k, v in summary_metrics.items():
        print("%10s, %0.3f" % (k, v))
