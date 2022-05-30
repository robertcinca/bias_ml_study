import warnings

import pandas as pd

from ml_bias_explainability.helpers import Helper, TextualExplanations, Visualizations

warnings.filterwarnings("ignore")  # "error", "ignore", "always", "default", "module" or "once"


class RunInteraction(object):
    """
    Builds an interactive explanation and action tool.
    """

    def __init__(
        self,
        csv_path,
        output,
        output_folder_location,
        verbose=False,
        minimum_sample_count=20,  # 0 to infinity
        minimum_delta_of_interest=0.04,  # 0.0 to 1.0
        minimum_sample_change_delta_of_interest=0.005,  # 0.0 to 1.0
        minimum_volatility_delta_of_interest=0.005,  # 0.0 to 1.0
    ):
        self.csv_path = csv_path
        self.output = output
        self.verbose = verbose
        self.output_folder_location = output_folder_location
        self.visual_folder_location = f"{output_folder_location}/visualizations"

        # fixed numerical variables
        self.minimum_sample_count = minimum_sample_count
        self.minimum_delta_of_interest = minimum_delta_of_interest
        self.minimum_sample_change_delta_of_interest = minimum_sample_change_delta_of_interest
        self.minimum_volatility_delta_of_interest = minimum_volatility_delta_of_interest

    def main(self, original_df):
        print("Creating visualizations based on the computed datasets.\n")

        print("Creating correlation plot.\n")
        Visualizations(
            output_folder_location=self.output_folder_location, show=False, save_fig=True
        ).correlation_plot(original_df, 0.2)

        print("Creating delta bias plots.\n")
        self._explain_ml_model(original_df)

        print("Creating sensitivity analysis plots.\n")
        self._explain_sensitivity_analysis_prediction_change(original_df)
        self._explain_sensitivity_analysis_prediction_volatility(original_df)

        print("Visualization creation completed.\n")

    def _explain_ml_model(self, original_df):
        # print("\nLOG:: Analysis type: ml_models_evaluation.\n")

        columns_of_interest = [
            "equalized_odds",
            "equal_opportunity",
            "statistical_parity",
            "treatment_equality",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
        metrics_df = pd.read_csv(f"{self.output_folder_location}/ml_models_evaluation.csv")

        for feature in self._features_to_analyze(metrics_df, "feature"):

            metrics_feature_df = self._feature_df_to_analyze(metrics_df, feature, "feature")

            # Only analyze rows with enough samples
            metrics_feature_df = metrics_feature_df[
                metrics_feature_df.sample_count > self.minimum_sample_count
            ]

            # print(f"\nLOG:: Relative delta bias for feature: {feature}.\n")
            self._relative_delta_bias(original_df, metrics_feature_df, feature, columns_of_interest)

            # print(f"\nLOG:: Absolute delta bias for feature: {feature}.\n")
            self._absolute_delta_bias(original_df, metrics_feature_df, feature, columns_of_interest)

    def _explain_sensitivity_analysis_prediction_change(self, original_df):
        # print("\nLOG:: Analysis type: sensitivity_predictions_change.\n")

        metrics_df = self._remove_nans(
            pd.read_csv(f"{self.output_folder_location}/sensitivity_predictions_change.csv")
        )

        for feature in self._features_to_analyze(metrics_df, "column_name"):

            metrics_feature_df = self._feature_df_to_analyze(metrics_df, feature, "column_name")

            for feature_value in self._feature_values_to_analyze(metrics_feature_df):
                # print(
                #     f"\nLOG:: Observing the percentage samples changing for feature: {feature} and feature value: {feature_value}.\n"
                # )

                # Only keep feature value of interest
                metrics_feature_value_df = metrics_feature_df[
                    metrics_feature_df.value_original == feature_value
                ]

                self._measure_prediction_changes(
                    original_df, metrics_feature_value_df, feature, feature_value
                )

    def _explain_sensitivity_analysis_prediction_volatility(self, original_df):
        # print("\nLOG:: Analysis type: sensitivity_predictions_volatility.\n")

        metrics_df = self._remove_nans(
            pd.read_csv(f"{self.output_folder_location}/sensitivity_predictions_volatility.csv")
        )

        for feature in self._features_to_analyze(metrics_df, "column_name"):

            metrics_feature_df = self._feature_df_to_analyze(metrics_df, feature, "column_name")

            for feature_value in self._feature_values_to_analyze(metrics_feature_df):
                # print(
                #     f"\nLOG:: Observing the change in prediction probabilities for feature: {feature} and feature value: {feature_value}.\n"
                # )

                # Only keep feature value of interest
                metrics_feature_value_df = metrics_feature_df[
                    metrics_feature_df.value_original == feature_value
                ]

                self._measure_prediction_volatility(
                    original_df, metrics_feature_value_df, feature, feature_value
                )

    def _remove_nans(self, metrics_df):
        # Ensure validity of inputs – No NaNs
        object_list, numeric_list = Helper.get_df_types_list(metrics_df)
        for column in object_list:
            metrics_df[column].fillna("", inplace=True)
        for column in numeric_list:
            metrics_df[column].fillna(-1, inplace=True)
        return metrics_df

    def _features_to_analyze(self, metrics_df, column_name):
        """Evaluate feature by feature"""
        return metrics_df[column_name].unique()

    def _feature_df_to_analyze(self, metrics_df, feature, column_name):
        """Filter df by feature"""
        return metrics_df[metrics_df[column_name] == feature]

    def _feature_values_to_analyze(self, metrics_feature_df):
        """Evaluate feature value by feature value"""
        return metrics_feature_df.value_original.unique()

    def _relative_delta_bias(self, original_df, chosen_feature_df, feature, columns_of_interest):
        """
        Compare differences in feature value metrics by training type
        E.g., feature: sex, feature values: male, female.
        Least biased = smallest difference in the max range of each feature value evaluation metric
        """
        grouped_df = pd.DataFrame()
        delta_columns_of_interest = [f"{column}_delta" for column in columns_of_interest]

        for column in columns_of_interest:
            grouped_df[f"{column}_delta"] = chosen_feature_df.groupby("training_type")[
                column
            ].apply(lambda g: g.max() - g.min())

        grouped_df["training_type"] = grouped_df.index

        if not grouped_df.empty:
            # Textual Explanations
            for column in columns_of_interest:
                min_value = grouped_df[f"{column}_delta"].min()

                original_value = grouped_df.loc["original_dataset", f"{column}_delta"]
                delta_value = original_value - min_value

                if delta_value > self.minimum_delta_of_interest and self.verbose:
                    min_value_index = grouped_df[f"{column}_delta"].idxmin()
                    action_to_perform = grouped_df.loc[min_value_index, "training_type"]

                    TextualExplanations().relative_delta_bias(
                        feature, action_to_perform, column, delta_value, original_value, min_value
                    )

            if delta_value > 0:
                # Visual Explanations
                figure_title = (
                    f"Max delta in feature value metrics for feature '{feature}' by training type"
                )
                y_label = "delta"

                Visualizations(
                    output_folder_location=f"{self.visual_folder_location}/analysis_2_and_3_ml_models/analysis_2_relative_delta_bias",
                    show=False,
                    save_fig=True,
                    figure_title=figure_title,
                ).stacked_visualization(grouped_df, delta_columns_of_interest, y_label, feature)

                Visualizations(
                    output_folder_location=f"{self.visual_folder_location}/analysis_2_and_3_ml_models/analysis_2_relative_delta_bias",
                    show=False,
                    save_fig=True,
                    figure_title=figure_title,
                ).small_multiples_plot(grouped_df, delta_columns_of_interest, y_label, feature)

    def _absolute_delta_bias(self, original_df, chosen_feature_df, feature, columns_of_interest):
        """
        Look at the effect of training type on the same feature value
        E.g., feature: sex, feature value: female.
        Least biased = highest values for evaluation metrics
        """
        delta_columns_of_interest = [f"{column}_delta" for column in columns_of_interest]
        values_grouped_obj = chosen_feature_df.groupby("feature_value")

        for value_grouped_obj in values_grouped_obj:
            feature_value = Helper.strip_text_for_file_naming(value_grouped_obj[0])
            grouped_df = value_grouped_obj[1]

            # Get the balanced data output (it's the one we want to compare)
            if "original_dataset" in grouped_df.training_type.values:
                balanced_row = grouped_df[grouped_df.training_type == "original_dataset"]

                # Textual Explanations
                for column in columns_of_interest:
                    grouped_df[f"{column}_delta"] = grouped_df[column] - balanced_row[column].values

                    max_value = grouped_df[f"{column}_delta"].max()
                    if max_value > self.minimum_delta_of_interest and self.verbose:
                        max_value_index = grouped_df[f"{column}_delta"].idxmax()
                        action_to_perform = grouped_df.loc[max_value_index, "training_type"]

                        TextualExplanations().absolute_delta_bias(
                            feature,
                            action_to_perform,
                            column,
                            feature_value,
                            max_value,
                            balanced_row,
                            grouped_df,
                            max_value_index,
                        )

                # Visual Explanations
                background_color = True
                figure_title = f"Absolute delta in feature value metrics for feature '{feature}' and feature value '{feature_value}' by training type"
                y_label = "delta"

                Visualizations(
                    output_folder_location=f"{self.visual_folder_location}/analysis_2_and_3_ml_models/analysis_3_absolute_delta_bias",
                    show=False,
                    save_fig=True,
                    figure_title=figure_title,
                ).stacked_visualization(
                    grouped_df,
                    delta_columns_of_interest,
                    y_label,
                    feature,
                    feature_value,
                    background_color,
                )

                Visualizations(
                    output_folder_location=f"{self.visual_folder_location}/analysis_2_and_3_ml_models/analysis_3_absolute_delta_bias",
                    show=False,
                    save_fig=True,
                    figure_title=figure_title,
                ).small_multiples_plot(
                    grouped_df,
                    delta_columns_of_interest,
                    y_label,
                    feature,
                    feature_value,
                    background_color,
                )

    def _measure_prediction_changes(
        self, original_df, metrics_feature_value_df, feature, feature_value
    ):
        """
        Measuring bias in sensitivity analysis: % samples where predicted output changes
        """
        feature_value = Helper.strip_text_for_file_naming(feature_value)

        # Get data where the prediction changes
        changed_predictions_df = metrics_feature_value_df[
            (
                metrics_feature_value_df.label_predicted_original
                != metrics_feature_value_df.label_predicted_new
            )
        ]

        # Create new columns to be able to calculate change ratio
        changed_predictions_df["changed_sample_size"] = changed_predictions_df["size"]
        metrics_feature_value_df["changed_sample_size"] = 0

        # If there are missing training types, add them in
        # (as it means that training type yields no prediction changes, which is good)
        for value in metrics_feature_value_df.training_type.unique():
            if value not in changed_predictions_df.training_type.unique():
                changed_predictions_df = changed_predictions_df.append(
                    metrics_feature_value_df[metrics_feature_value_df.training_type == value].iloc[
                        0
                    ]
                )

        # Compute total sample size of each batch
        changed_predictions_df["total_sample_size"] = changed_predictions_df.apply(
            lambda row: metrics_feature_value_df[
                (metrics_feature_value_df.column_name == row["column_name"])
                & (metrics_feature_value_df.value_original == row["value_original"])
                & (metrics_feature_value_df.value_new == row["value_new"])
                & (metrics_feature_value_df.training_type == row["training_type"])
            ]["size"].sum(),
            axis=1,
        )

        changed_predictions_df["prediction_change_ratio"] = (
            changed_predictions_df["changed_sample_size"]
            / changed_predictions_df["total_sample_size"]
        )

        # Group by feature, all perturbations of the feature values
        grouped_df = pd.DataFrame()
        grouped_df["mean_prediction_change_ratio"] = changed_predictions_df.groupby(
            ["training_type"]
        )["prediction_change_ratio"].mean()
        grouped_df["sum_size"] = changed_predictions_df.groupby(["training_type"])["size"].sum()
        grouped_df["total_sample_size"] = changed_predictions_df.groupby(["training_type"])[
            "total_sample_size"
        ].max()
        grouped_df["column_name"] = changed_predictions_df.groupby(["training_type"])[
            "column_name"
        ].max()
        grouped_df["training_type"] = grouped_df.index

        min_value = grouped_df["mean_prediction_change_ratio"].min()

        if not grouped_df.empty and grouped_df.index.isin(["original_dataset"]).any():
            original_value = grouped_df.loc["original_dataset", "mean_prediction_change_ratio"]
            delta_value = original_value - min_value

            # Textual Explanations
            if delta_value > self.minimum_sample_change_delta_of_interest and self.verbose:
                min_value_index = grouped_df["mean_prediction_change_ratio"].idxmin()
                action_to_perform = grouped_df.loc[min_value_index, "training_type"]

                TextualExplanations().sensitivity_prediction_change_decrease(
                    feature,
                    action_to_perform,
                    feature_value,
                    delta_value,
                    original_value,
                    min_value,
                )

            # Visual Explanations
            figure_title = f"Ratio of samples changing output for feature '{feature}' and feature value '{feature_value}' by training type"
            columns_of_interest = ["mean_prediction_change_ratio"]
            y_label = "ratio"

            # Only stacked visualization makes sense (only 1 var)
            Visualizations(
                output_folder_location=f"{self.visual_folder_location}/analysis_4_and_5_sensitivity/analysis_4_prediction_change",
                show=False,
                save_fig=True,
                figure_title=figure_title,
            ).stacked_visualization(
                grouped_df, columns_of_interest, y_label, feature, feature_value
            )

    def _measure_prediction_volatility(
        self, original_df, metrics_feature_value_df, feature, feature_value
    ):
        """
        Measuring bias in sensitivity analysis: volatility in the predicted output
        """
        feature_value = Helper.strip_text_for_file_naming(feature_value)

        # Group by feature, all perturbations of the feature values
        grouped_df = metrics_feature_value_df.groupby(["training_type"]).agg(
            mean_standard_deviation=("standard_deviation", "mean"),
        )
        grouped_df["training_type"] = grouped_df.index

        if not grouped_df.empty and grouped_df.index.isin(["original_dataset"]).any():
            min_value = grouped_df["mean_standard_deviation"].min()
            original_value = grouped_df.loc["original_dataset", "mean_standard_deviation"]
            delta_value = original_value - min_value

            # Textual Explanations
            if delta_value > self.minimum_volatility_delta_of_interest and self.verbose:
                min_value_index = grouped_df["mean_standard_deviation"].idxmin()
                action_to_perform = grouped_df.loc[min_value_index, "training_type"]

                TextualExplanations().sensitivity_prediction_volatility_decrease(
                    feature,
                    action_to_perform,
                    feature_value,
                    delta_value,
                    original_value,
                    min_value,
                )

            # Visual Explanations
            figure_title = f"Volatility of samples' output for feature '{feature}' and feature value '{feature_value}' by training type"
            columns_of_interest = ["mean_standard_deviation"]
            y_label = "volatility"

            # Only stacked visualization makes sense (only 1 var)
            Visualizations(
                output_folder_location=f"{self.visual_folder_location}/analysis_4_and_5_sensitivity/analysis_5_prediction_volatility",
                show=False,
                save_fig=True,
                figure_title=figure_title,
            ).stacked_visualization(
                grouped_df, columns_of_interest, y_label, feature, feature_value
            )
