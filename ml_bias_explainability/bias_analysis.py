"""
THIS CLASS IS DEPRECATED.
"""

from collections import Counter
from math import exp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from ml_bias_explainability.helpers import Visualizations


class BiasAnalysis:
    """
    An API class designed to analyze a dataset for bias potential.
    Based on the ProPublica analysis of recidivism.
    """

    def run_pipeline(
        self,
        df,
        input_features,
        output_feature,
        binary_output_feature,
        correlation_plot_threshold=0.5,
        minimum_unbalanced_ratio=0.1,
        unique_values_threshold=10,
        significant_output_threshold=0.2,
    ):
        print("A list of correlations between features of the dataset in descending order.")
        BiasAnalysis().check_correlation_of_features_text(df)

        print("A heatmap of all correlations above a defined threshold.")
        Visualizations(output_folder_location="", show=True, save_fig=False).correlation_plot(
            df, correlation_plot_threshold
        )

        print("A breakdown of unbalanced occurrence of elements by feature.")
        BiasAnalysis().check_feature_inbalance(df, minimum_unbalanced_ratio)

        print("A plot of an input feature against the output feature.")
        BiasAnalysis().check_output_inbalance(df, output_feature, unique_values_threshold)

        print(
            "A logistic regression analysis of differences in output between different feature values."
        )
        BiasAnalysis().output_significance_by_score(
            df, input_features, binary_output_feature, significant_output_threshold
        )
        return "Pipeline has finished running."

    def check_correlation_of_features_text(self, df):
        """
        Returns a list of correlations between all features of the
        dataset in descending order.
        """
        correlation_matrix = df.corr().abs()

        correlations = (
            correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool)
            )
            .stack()
            .sort_values(ascending=False)
        )

        relevant_correlations = correlations[correlations != 1.000]

        print(relevant_correlations[: min(10, len(relevant_correlations.index))])
        return relevant_correlations

    def check_feature_inbalance(self, df, minimum_unbalanced_ratio=0.1):
        """
        Returns a breakdown on most common vs least common elements for each feature in a dataframe.
        """
        for column in df:
            column_counter = Counter(df[column])

            if len(column_counter) > 1:
                least_common_element = column_counter.most_common()[-1]
                most_common_element = column_counter.most_common()[0]
                occurence_ratio = least_common_element[1] / most_common_element[1]

                if occurence_ratio < minimum_unbalanced_ratio:
                    print(
                        f"""
WARNING: column {column} might be unbalanced.

The most common value occurs more than the least common column by a factor of {1/occurence_ratio}.

The most common element is: {most_common_element}
The least common element is: {least_common_element}
"""
                    )

    def check_output_inbalance(
        self,
        df,
        output_feature,
        unique_values_threshold=10,
    ):
        """
        Plots a distribution of specific input features vs the output feature.
        """
        for column in df:
            # Only plot where at least 2 unique values up to a defined threshold
            if 1 < df[column].nunique() <= unique_values_threshold:
                pd.crosstab(df[output_feature], df[column]).plot.bar(figsize=(20, 10), fontsize=20)
                plt.show()

    def output_significance_by_score(
        self,
        df,
        input_features,
        binary_output_feature,
        significant_output_threshold=0.2,
    ):
        """
        A logistic regression analysis of differences in output between different feature values.
        """
        input_formula = ""
        for column in input_features:
            input_formula = f"{input_formula} + {column}"

        result = sm.formula.glm(
            f"{binary_output_feature} ~ {input_formula}", family=sm.families.Binomial(), data=df
        ).fit()
        print(result.summary())

        results_df = result.summary2().tables[1]
        intercept_coefficient = results_df.at["Intercept", "Coef."]
        control = exp(intercept_coefficient) / (1 + exp(intercept_coefficient))

        for i, coefficient_value in enumerate(results_df["Coef."]):
            if i > 0:
                probability = exp(coefficient_value) / (
                    1 - control + (control * exp(coefficient_value))
                )
                if (
                    probability > 1 + significant_output_threshold
                    or probability < 1 - significant_output_threshold
                ):
                    print(
                        f"""
                    Samples that contain {results_df.index[i]} are {probability:.2f} likely to receive a higher score
                    compared to samples that don't contain {results_df.index[i]}.
                    """
                    )

        # https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe
        significant_variables = list(results_df[results_df["P>|z|"] <= 0.05].index)[1:]
        print(f"The significant variables are: {significant_variables}")

        return results_df

    def check_probability_of_outcomes(
        self,
        model,
        x_test,
        y_test,
        probability_difference_threshold=0.2,
    ):
        predictions = model.predict(x_test)

        if len(predictions[0]) > 1:  # Need to have at least 2 output probabilities
            for index, prediction in enumerate(predictions):

                sorted_predictions = sorted(prediction[::-1], reverse=True)
                sorted_indexes = np.argsort(-prediction)

                probability_difference = sorted_predictions[0] - sorted_predictions[1]

                if probability_difference < probability_difference_threshold:
                    print(
                        f"""
WARNING: prediction {index} might need further analysis.

The most likely outcome of {sorted_indexes[0]} has a probability of {sorted_predictions[0]:.3f}.
However, the second most likely outcome of {sorted_indexes[1]} has a probability of {sorted_predictions[1]:.3f}.

The difference in probabilities between these two potentially different outcomes is: {probability_difference:.3f}
"""
                    )

                    plt.plot(prediction, "bo", label="Training loss")
                    plt.title(f"Prediction probabilities for prediction {index}")
                    plt.xlabel("Output Label")
                    plt.ylabel("Probability")
                    plt.legend()
                    plt.ylim(top=1)
                    plt.ylim(bottom=0)
                    plt.show()
        else:
            print("Each prediction contains fewer than 2 output probabilities.")
