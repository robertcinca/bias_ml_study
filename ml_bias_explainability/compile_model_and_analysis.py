import pandas as pd

from ml_bias_explainability import (
    ComputeEvaluationMetric,
    CreateEncoding,
    ExplainBias,
    FindBestParameter,
    GenerateModel,
)
from ml_bias_explainability.helpers import Helper


class CompileModelAndAnalysis(object):
    """
    Builds a Neural Network by tuning the hyperparameter.

    It then runs a Sensitivity Analysis by tweaking the values of features
    and observing the effect on the output.
    """

    def __init__(
        self,
        training_types,
        features_of_interest,
        output,
        columns_to_remove,
        csv_path,
        param_grid,
        unique_values=10,
        best_params=None,
    ):
        self.output = output
        self.columns_to_remove = columns_to_remove
        self.param_grid = param_grid
        self.csv_path = csv_path
        self.unique_values = unique_values

        self.training_types = training_types or ["original_dataset"]
        self.features_of_interest = features_of_interest or ["all"]
        self.best_params = best_params
        self.evaluation_columns = [
            "training_type",
            "feature",
            "feature_value",
            "sample_count",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "equalized_odds",
            "equal_opportunity",
            "statistical_parity",
            "treatment_equality",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]

    def main(self, df):
        (
            evaluation_rows_of_rows,
            raw_predictions_df_list,
            predictions_change_df_list,
            volatility_df_list,
            sensitivity_evaluation_df_list,
        ) = self._run_pipeline(df)

        # Print and store outputs of the ML models and the sensitivity analysis
        model_evaluation_df = pd.DataFrame(
            [item for sublist in evaluation_rows_of_rows for item in sublist],
            columns=self.evaluation_columns,
        )
        raw_predictions_df = pd.concat(raw_predictions_df_list)
        predictions_change_df = pd.concat(predictions_change_df_list)
        volatility_df = pd.concat(volatility_df_list)
        sensitivity_evaluation_df = pd.concat(sensitivity_evaluation_df_list)

        print("\nData Computation Phase Completed Successfully.\n")
        return (
            model_evaluation_df,
            raw_predictions_df,
            predictions_change_df,
            volatility_df,
            sensitivity_evaluation_df,
        )

    def _run_pipeline(self, df_original):
        sensitivity_evaluation_df_list, raw_predictions_df_list = [], []
        predictions_change_df_list, volatility_df_list = [], []
        evaluation_rows_of_rows = []

        # define output
        df_original, self.output = Helper.define_output(df_original, self.output)

        for training_type in self.training_types:
            if training_type == "remove_feature":

                features_to_remove = list(df_original.columns)
                features_to_remove.remove(self.output)

            else:
                features_to_remove = [None]

            for feature_to_remove in features_to_remove:

                df = Helper.remove_certain_features(df_original.copy(), feature_to_remove)

                print(
                    f"RUNNING ANALYSIS FOR TRAINING TYPE: '{training_type}_{feature_to_remove}'\n"
                    if feature_to_remove
                    else f"RUNNING ANALYSIS FOR TRAINING TYPE: '{training_type}'\n"
                )

                (
                    partial_evaluation_rows_of_rows,
                    partial_raw_predictions_df_list,
                    partial_predictions_change_df_list,
                    partial_volatility_df_list,
                    partial_sensitivity_evaluation_df_list,
                ) = self._run_pipeline_using_training_type(df, training_type, feature_to_remove)

                sensitivity_evaluation_df_list = (
                    sensitivity_evaluation_df_list + partial_sensitivity_evaluation_df_list
                )
                raw_predictions_df_list = raw_predictions_df_list + partial_raw_predictions_df_list
                predictions_change_df_list = (
                    predictions_change_df_list + partial_predictions_change_df_list
                )
                volatility_df_list = volatility_df_list + partial_volatility_df_list
                evaluation_rows_of_rows = evaluation_rows_of_rows + partial_evaluation_rows_of_rows

        return (
            evaluation_rows_of_rows,
            raw_predictions_df_list,
            predictions_change_df_list,
            volatility_df_list,
            sensitivity_evaluation_df_list,
        )

    def _run_pipeline_using_training_type(self, df, training_type, feature_to_remove=None):
        partial_sensitivity_evaluation_df_list, partial_raw_predictions_df_list = [], []
        partial_predictions_change_df_list, partial_volatility_df_list = [], []
        partial_evaluation_rows_of_rows = []

        numeric_list, object_list = Helper.get_input_column_dtypes(df)

        # Get best parameters for model design if not provided
        if not self.best_params:
            self.best_params = FindBestParameter(self.output, self.param_grid).main(df)

        df_ds, output_size = Helper.df_to_dataset(df)
        train_ds, val_ds, test_ds = Helper.split_ds_into_sets(
            df_ds, batch_size=self.best_params["batch_size"]
        )

        encoded_features, all_inputs = CreateEncoding().main(numeric_list, object_list, train_ds)

        model = GenerateModel(self.best_params, output_size).main(
            encoded_features, all_inputs, train_ds, val_ds, test_ds
        )

        evaluation_rows, _ = ComputeEvaluationMetric(
            self.features_of_interest,
            self.best_params["batch_size"],
            training_type,
            feature_to_remove,
        ).main(model, test_ds)

        # Run sensitivity analysis
        (
            raw_predictions_partial_df,
            predictions_change_partial_df,
            volatility_partial_df,
            sensitivity_evaluation_partial_df,
        ) = ExplainBias(
            training_type,
            self.features_of_interest,
            self.output,
            self.best_params["batch_size"],
            self.unique_values,
            self.evaluation_columns,
            feature_to_remove,
        ).sensitivity_analysis(
            model, test_ds
        )

        partial_evaluation_rows_of_rows.append(evaluation_rows)
        partial_raw_predictions_df_list.append(raw_predictions_partial_df)
        partial_predictions_change_df_list.append(predictions_change_partial_df)
        partial_volatility_df_list.append(volatility_partial_df)
        partial_sensitivity_evaluation_df_list.append(sensitivity_evaluation_partial_df)

        return (
            partial_evaluation_rows_of_rows,
            partial_raw_predictions_df_list,
            partial_predictions_change_df_list,
            partial_volatility_df_list,
            partial_sensitivity_evaluation_df_list,
        )
