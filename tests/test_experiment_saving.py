from gnm.fitting.experiment_saving import *
from gnm.fitting.experiment_dataclasses import Experiment

eval = ExperimentEvaluation()
dummy_code = Experiment(None, None, None, None)
eval.save_experiment(dummy_code)
