# pylama:ignore=W0611

import os
import warnings

import tensorflow as tf

from ml_bias_explainability.compute_evaluation_metric import ComputeEvaluationMetric
from ml_bias_explainability.create_encoding import CreateEncoding
from ml_bias_explainability.explain_bias import ExplainBias
from ml_bias_explainability.find_best_parameters import FindBestParameter
from ml_bias_explainability.generate_model import GenerateModel

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")  # Modify output level of tf model building
warnings.filterwarnings("ignore")  # "error", "ignore", "always", "default", "module" or "once"
