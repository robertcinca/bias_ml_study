import pandas as pd
from joblib import Parallel, delayed

from ml_bias_explainability import ComputeEvaluationMetric
from ml_bias_explainability.helpers import Helper


class ExplainBias:
    """An API class designed to explain model bias"""

    def __init__(
        self,
        training_type,
        features_of_interest,
        output,
        batch_size,
        unique_values,
        evaluation_columns,
        feature_to_remove=None,
    ):
        self.training_type = training_type
        self.features_of_interest = features_of_interest
        self.output = output
        self.batch_size = batch_size
        self.unique_values = unique_values
        self.evaluation_columns = evaluation_columns
        self.feature_to_remove = feature_to_remove

    def sensitivity_analysis(
        self,
        model,
        test_ds,
    ):
        # get predictions for original dataset
        predictions_old = model.predict(test_ds)

        unbatched_test_ds = test_ds.unbatch()
        test_df = Helper.extract_df_from_ds(unbatched_test_ds)

        # New sensitivity analysis: only on features of interest
        feature_list = self._features_for_sensitivity_analysis()

        print("\nRunning the sensitivity analysis feature by feature. This might take a while.")

        returned_data = Parallel(n_jobs=-1, prefer="threads", verbose=8)(
            delayed(self._change_prediction_by_feature)(feature, model, predictions_old, test_df)
            for feature in feature_list
        )

        list_of_predictions_dfs = [item[0] for item in returned_data]
        list_of_evaluation_dfs = [item[1] for item in returned_data]

        raw_predictions_df = pd.concat(list_of_predictions_dfs)
        predictions_change_df = self._measure_prediction_changes(raw_predictions_df)
        volatility_df = self._measure_volatility(raw_predictions_df)
        sensitivity_evaluation_df = pd.concat(list_of_evaluation_dfs)

        training_type_value = (
            f"{self.training_type}_{self.feature_to_remove}"
            if self.feature_to_remove
            else self.training_type
        )

        raw_predictions_df.insert(loc=0, column="training_type", value=training_type_value)
        predictions_change_df.insert(loc=0, column="training_type", value=training_type_value)
        volatility_df.insert(loc=0, column="training_type", value=training_type_value)

        return (
            raw_predictions_df,
            predictions_change_df,
            volatility_df,
            sensitivity_evaluation_df,
        )

    def _change_prediction_by_feature(self, feature, model, predictions_old, test_df):
        column_predictions_df_list, evaluation_df_list = [], []

        # if column is numeric, perturbe value, otherwise cycle through values
        is_numeric = False
        if test_df[feature].dtype.kind in "biufc":
            is_numeric = True
            max_col_val = test_df[feature].max()
            min_col_val = test_df[feature].min()
            step = max_col_val - min_col_val  # normalize over values range
            column_values = [-step * 0.10, -step * 0.05, step * 0.05, step * 0.10]  # +-5%, +-10%

            if test_df[feature].dtype.kind in "biu":  # keep 'count' features as int
                column_values = [round(x, 0) for x in column_values]
        else:
            column_values = test_df[feature].unique()

        for column_value in column_values:
            modified_df = test_df.copy()
            if is_numeric:
                value_original = "x"
                value_new = f"{column_value} from the original"
                modified_df[feature] = modified_df[feature].apply(
                    lambda x: round(max(min(x + column_value, max_col_val), min_col_val), 2)
                )
            else:
                value_original = test_df.loc[test_df.index.tolist()][feature].to_numpy()
                id_value = test_df[feature].eq(column_value).idxmax()
                value_new = test_df.loc[id_value][feature]
                modified_df[feature] = column_value

            unbatched_modified_test_ds, _ = Helper.df_to_dataset(modified_df)

            evaluation_rows, predictions_new = ComputeEvaluationMetric(
                self.features_of_interest,
                self.batch_size,
                self.training_type,
                self.feature_to_remove,
            ).main(model, unbatched_modified_test_ds.batch(self.batch_size))

            sensitivity_evaluation_partial_df = pd.DataFrame(
                evaluation_rows, columns=self.evaluation_columns
            )
            sensitivity_evaluation_partial_df.insert(3, "column_name", feature)
            sensitivity_evaluation_partial_df.insert(4, "column_value", value_new)

            labels_predicted_original = [int("%.0f" % elem) for elem in predictions_old]
            labels_predicted_new = [int("%.0f" % elem) for elem in predictions_new]

            prediction_difference_array = predictions_new - predictions_old
            abs_prediction_difference_array = [
                abs(x) for x in prediction_difference_array.flatten()
            ]

            toward_label_array = [0 if el < 0 else 1 for el in prediction_difference_array]

            column_predictions_partial_df = pd.DataFrame(
                {
                    "column_name": feature,
                    "value_original": value_original,
                    "value_new": value_new,
                    "label_predicted_original": labels_predicted_original,
                    "label_predicted_new": labels_predicted_new,
                    "standard_deviation": abs_prediction_difference_array,
                    "standard_deviation_direction": toward_label_array,
                },
            )

            column_predictions_df_list.append(column_predictions_partial_df)
            evaluation_df_list.append(sensitivity_evaluation_partial_df)

        return [pd.concat(column_predictions_df_list), pd.concat(evaluation_df_list)]

    def _measure_prediction_changes(self, raw_predictions_df):
        """
        Measuring bias in sensitivity analysis: % samples where predicted output changes
        """
        aggregate_predictions_df = (
            raw_predictions_df.groupby(
                [
                    "column_name",
                    "value_original",
                    "value_new",
                    "label_predicted_original",
                    "label_predicted_new",
                ]
            )
            .size()
            .to_frame(name="size")
            .reset_index()
        )

        return aggregate_predictions_df

    def _measure_volatility(self, raw_predictions_df):
        """
        Measuring bias in sensitivity analysis: average volatility using mean of
        standard deviations of the difference between original and new prediction.
        """
        volatility_df = raw_predictions_df[
            raw_predictions_df["value_original"] != raw_predictions_df["value_new"]
        ]

        return (
            volatility_df.groupby(["column_name", "value_original", "value_new"])
            .agg(
                standard_deviation=("standard_deviation", "mean"),
                standard_deviation_direction=(
                    "standard_deviation_direction",
                    lambda x: x.value_counts().index[0],
                ),
            )
            .reset_index()
        )

    def _features_for_sensitivity_analysis(self):
        """
        Get columns for sensitivity analysis. All numeric + non-numeric cols below threshold
        """
        feature_list = self.features_of_interest.copy()
        if "all" in feature_list:
            feature_list.remove("all")
        if self.feature_to_remove in feature_list:
            feature_list.remove(self.feature_to_remove)

        return feature_list
