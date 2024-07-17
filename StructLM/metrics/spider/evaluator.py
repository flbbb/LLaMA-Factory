# encoding=utf8

import glob
import os

from .spider_exact_match import compute_exact_match_metric
from .spider_test_suite import compute_test_suite_metric


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):

        # find the path in data/downloads/extracted/*/spider
        DATA_PATH = os.environ["DATA_PATH"]
        matching_paths = glob.glob(f"{DATA_PATH}/downloads/extracted/*/spider/database")
        assert len(matching_paths) == 1
        db_dir = matching_paths[0]
        golds[0]["db_path"] = db_dir

        if self.args.seq2seq.target_with_db_id:
            # Remove database id from all predictions
            preds = [pred.split("|", 1)[-1].strip() for pred in preds]
        exact_match = compute_exact_match_metric(preds, golds)
        test_suite = compute_test_suite_metric(preds, golds, db_dir=db_dir)
        return {**exact_match, **test_suite}
