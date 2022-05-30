class TextualExplanations:
    def relative_delta_bias(
        self, feature, action_to_perform, column, delta_value, original_value, min_value
    ):
        print(
            f"""
===================================
Explanation II: Relative Delta Bias
===================================

For feature: {feature}
Performing action: {action_to_perform}
Leads to a decrease in the range of '{column}' for feature values of {round(delta_value,2)}

==========
Background
==========

The original range in feature values for training type "all_features"
And for the metric '{column}' was: {round(original_value,2)}
The new range after performing action {action_to_perform} would be {round(min_value,2)}

Reminder: The least biased option is the action that results in the smallest range in the evaluation metrics of the feature values.

"""
        )

    def absolute_delta_bias(
        self,
        feature,
        action_to_perform,
        column,
        feature_value,
        max_value,
        balanced_row,
        grouped_df,
        max_value_index,
    ):
        print(
            f"""
====================================
Explanation III: Absolute Delta Bias
====================================

For feature: {feature}
And feature value: {feature_value}
Performing action: {action_to_perform}
Leads to an increase in the metric '{column}' of {round(max_value,2)}

==========
Background
==========

The original value for training type: {balanced_row.training_type.values[0]}
And for the metric '{column}' was: {balanced_row[column].values[0]}
The new value after performing action {action_to_perform} would be {grouped_df.loc[max_value_index, column]}

Reminder: The least biased option is the action that results in the highest value for the evaluation metric of the feature value.

"""
        )

    def sensitivity_prediction_change_decrease(
        self, feature, action_to_perform, feature_value, delta_value, original_value, min_value
    ):
        print(
            f"""
======================================================
Explanation IV: Sensitivity Analysis Prediction Change
======================================================

Type: Reducing percentage samples changing prediction for a feature by modifying training type.

When conducting sensitivity analysis for feature: {feature}
And feature value: {feature_value}
Performing action: {action_to_perform}
Leads to a decrease in the ratio of samples that change predicted outcome by: {round(delta_value, 3)}

==========
Background
==========

The original mean prediction change ratio for training type {"all_features"} was: {round(original_value,3)}
The new mean prediction change ratio after performing action {action_to_perform} would be {round(min_value,3)}

Reminder: The least biased option is the action that results in the fewest ratio of samples changing when conducting sensitivity analysis on sensitive features

"""
        )

    def sensitivity_prediction_volatility_decrease(
        self, feature, action_to_perform, feature_value, delta_value, original_value, min_value
    ):
        print(
            f"""
=========================================================
Explanation V: Sensitivity Analysis Prediction Volatility
=========================================================

Type: Reducing the volatility in output prediction probabilities when conducting sensitivity analysis.

For feature: {feature}
And feature value: {feature_value}
Performing action: {action_to_perform}
Leads to a decrease in the standard deviation (volatility) of predicted output probabilities by: {round(delta_value, 3)}

==========
Background
==========

The original range in feature values for training type: {"all_features"}
And for standard deviation was: {round(original_value,2)}
The new range after performing action {action_to_perform} would be {round(min_value,2)}

Reminder: The least biased option is the action that results in the smallest change in output prediction probabilities when conducting sensitivity analysis.

"""
        )
