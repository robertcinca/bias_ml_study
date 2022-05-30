import numpy as np
import tensorflow as tf

from ml_bias_explainability.helpers import Helper


class ComputeEvaluationMetric:
    def __init__(self, features_of_interest, batch_size, training_type, feature_to_remove=None):
        self.features_of_interest = features_of_interest
        self.batch_size = batch_size
        self.training_type = training_type
        self.feature_to_remove = feature_to_remove

    def main(self, model, test_ds):
        evaluation_rows = []

        probabilities_predicted = model.predict(test_ds)
        model.predict(test_ds, workers=tf.data.AUTOTUNE, use_multiprocessing=True)

        labels_predicted = np.asarray(
            [int("%.0f" % elem) for elem in probabilities_predicted], dtype=np.int64
        )

        unbatched_test_ds = test_ds.unbatch()
        predicted_test_df = Helper.extract_df_from_ds(unbatched_test_ds)
        predicted_test_df["predicted"] = labels_predicted

        for feature in self.features_of_interest:
            if feature == self.feature_to_remove:
                continue

            if feature == "all":
                feature_values_raw = ["all"]
            else:
                feature_values_raw = predicted_test_df[feature].unique()

            for feature_value_raw in feature_values_raw:

                if feature_value_raw == "all":
                    partial_labels_actual = predicted_test_df["target"].to_numpy()
                    partial_labels_predicted = predicted_test_df["predicted"].to_numpy()
                else:
                    partial_labels_actual = predicted_test_df[
                        predicted_test_df[feature] == feature_value_raw
                    ]["target"].to_numpy()

                    partial_labels_predicted = predicted_test_df[
                        predicted_test_df[feature] == feature_value_raw
                    ]["predicted"].to_numpy()

                evaluation_headers_json = {
                    "training_type": (
                        f"{self.training_type}_{self.feature_to_remove}"
                        if self.feature_to_remove
                        else self.training_type
                    ),
                    "feature": feature,
                    "feature_value": feature_value_raw,
                }

                evaluation_metrics_json = self._compute_evaluation_metrics(
                    partial_labels_actual, partial_labels_predicted
                )

                evaluation_json = {**evaluation_headers_json, **evaluation_metrics_json}

                evaluation_rows.append(evaluation_json)

        return evaluation_rows, probabilities_predicted

    def _compute_evaluation_metrics(self, labels_actual, labels_predicted):
        tp = sum(labels_actual & labels_predicted)
        fp = sum(1 - labels_actual & labels_predicted)
        tn = sum(1 - labels_actual & 1 - labels_predicted)
        fn = sum(labels_actual & 1 - labels_predicted)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        sample_count = tp + tn + fp + fn

        if (precision + recall) == 0:
            f1_score = 0
        else:
            f1_score = 2 * ((precision * recall) / (precision + recall))

        equalized_odds = Helper.equalized_odds(tp, fp, tn, fn)

        equal_opportunity = Helper.equal_opportunity(
            tp,
            fp,
            tn,
            fn,
        )

        statistical_parity = Helper.statistical_parity(
            tp,
            fp,
            tn,
            fn,
        )

        treatment_equality = Helper.treatment_equality(
            tp,
            fp,
            tn,
            fn,
        )

        evaluation_metrics_json = {
            "sample_count": sample_count,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "equalized_odds": equalized_odds,  # definition 1
            "equal_opportunity": equal_opportunity,  # definition 2
            "statistical_parity": statistical_parity,  # definition 3
            "treatment_equality": treatment_equality,  # definition 6
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
        }

        return evaluation_metrics_json
