#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
import copy
import os
import traceback

import numpy as np
import utils.tool
from tqdm import tqdm
from utils.configure import Configure


class EvaluateTool(object):
    """
    The meta evaluator
    """

    def __init__(self, meta_args):
        self.meta_args = meta_args

    def evaluate(self, preds, golds, section):
        meta_args = self.meta_args
        summary = {}
        wait_for_eval = {}

        for pred, gold in zip(preds, golds):
            if gold["arg_path"] not in wait_for_eval.keys():
                wait_for_eval[gold["arg_path"]] = {"preds": [], "golds": []}
            wait_for_eval[gold["arg_path"]]["preds"].append(pred)
            wait_for_eval[gold["arg_path"]]["golds"].append(gold)

        lst = [
            (arg_path, preds_golds) for arg_path, preds_golds in wait_for_eval.items()
        ]
        print([arg_path for arg_path, preds_golds in lst])
        cfg_to_ignore = ["bird", "spider", "sparc", "cosql", "multiwoz", "kvret"]
        for arg_path, preds_golds in tqdm(lst):
            for k in cfg_to_ignore:
                if k in arg_path:
                    continue

            print("Evaluating {}...".format(arg_path))
            args = Configure.refresh_args_by_file_cfg(
                os.path.join(meta_args.dir.configure, arg_path), meta_args
            )
            evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
            try:
                summary_tmp = evaluator.evaluate(
                    preds_golds["preds"], preds_golds["golds"], section
                )
            except Exception as e:
                print("Error in evaluating {}:".format(arg_path))
                print(str(e))
                # traceback.print_exc()
                summary_tmp = {"error": 0}
            print(summary_tmp)
            for key, metric in summary_tmp.items():  # TODO
                summary[os.path.join(arg_path, key)] = metric
            # summary[os.path.join(arg_path, args.train.stop)] = summary_tmp[args.train.stop]

        summary["avr"] = float(np.mean([float(v) for k, v in summary.items()]))
        return summary
