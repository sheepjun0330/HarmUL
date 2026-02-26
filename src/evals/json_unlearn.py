from evals.base import Evaluator


class JSONUnlearnEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("JSON_UNLEARN", eval_cfg, **kwargs)
