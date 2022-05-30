from time import time

import bootstrap  # noqa
import pandas as pd

# from ml_bias_explainability.find_best_parameters import FindBestParameter
from ml_bias_explainability.compile_model_and_analysis import CompileModelAndAnalysis
from ml_bias_explainability.helpers.helper import Helper
from ml_bias_explainability.run_tool_interaction import RunInteraction


def clean_dataset(csv_path, columns_to_remove):
    # The dataset used is taken from https://github.com/propublica/compas-analysis
    df = pd.read_csv(csv_path, index_col=0)

    # Based on propublica analysis, filter rows where COMPAS scoring and
    # arrest date weren't within 30 days of each other
    df = df[df.days_b_screening_arrest <= 30]
    df = df[df.days_b_screening_arrest >= -30]

    # get rid of columns that shouldn't be in the input model
    filtered_columns_to_remove = df.filter(columns_to_remove)
    df.drop(filtered_columns_to_remove, inplace=True, axis=1)

    # Ensure validity of inputs – No NaNs
    object_list, numeric_list = Helper.get_df_types_list(df)
    for column in object_list:
        df[column].fillna("", inplace=True)
    for column in numeric_list:
        df[column].fillna(-1, inplace=True)

    return df


if __name__ == "__main__":
    # STEP 0: Fill out initial variables
    training_types = [
        "original_dataset",
        "remove_feature",
    ]
    features_of_interest = ["all", "race", "sex", "age_cat"]
    output = "two_year_recid"
    columns_to_remove = Helper.read_yaml("scripts/data/1_compass_recidivism/columns_to_remove.yml")
    csv_path = "scripts/data/1_compass_recidivism/compas-scores-two-years.csv"
    output_folder_location = "scripts/data/1_compass_recidivism/output"
    unique_values = 6
    verbose = False

    # Param grid: either a yaml file with multiple options or hard coded
    # param_grid = Helper.read_yaml("scripts/data/1_compass_recidivism/param_grid.yml")
    param_grid = dict(
        batch_size=[128, 256],
        epochs=[15],
        optimizer=["RMSprop", "Adam", "Nadam"],
        init_mode=["he_uniform", "glorot_uniform"],
        activation=["relu", "sigmoid"],
        weight_constraint=[1, 2, 3],
        dropout_rate=[0.01, 0.1, 0.2],
        neurons=[32, 64, 128],
        hidden_layers=[1, 2, 3, 4],
    )

    # STEP 1: Get cleaned dataset
    df = clean_dataset(csv_path, columns_to_remove)

    # STEP 2 [OPT]: Get best parameters (either compile at run time or define)
    # best_params = FindBestParameter(output, param_grid).main(df)
    best_params = dict(
        batch_size=256,
        epochs=15,
        optimizer="Nadam",
        init_mode="he_uniform",
        activation="sigmoid",
        weight_constraint=1,
        dropout_rate=0.2,
        neurons=64,
        hidden_layers=2,
    )

    t0 = time()

    # STEP 3: Create and Run analysis on model
    (
        model_evaluation_df,
        raw_predictions_df,
        predictions_change_df,
        volatility_df,
        sensitivity_evaluation_df,
    ) = CompileModelAndAnalysis(
        training_types,
        features_of_interest,
        output,
        columns_to_remove,
        csv_path,
        param_grid,
        unique_values,
        best_params,
    ).main(
        df
    )

    print("%.4f sec" % (time() - t0))

    # Saving the computed datasets. These are used in the analysis part of the tool
    model_evaluation_df = model_evaluation_df.round(2)
    raw_predictions_df = raw_predictions_df.round(2)
    predictions_change_df = predictions_change_df.round(2)
    volatility_df = volatility_df.round(2)
    sensitivity_evaluation_df = sensitivity_evaluation_df.round(2)

    model_evaluation_df.to_csv(
        f"{output_folder_location}/ml_models_evaluation.csv",
        index=False,
    )
    raw_predictions_df.to_csv(
        f"{output_folder_location}/sensitivity_predictions_raw.csv",
        index=False,
    )
    predictions_change_df.to_csv(
        f"{output_folder_location}/sensitivity_predictions_change.csv",
        index=False,
    )
    volatility_df.to_csv(
        f"{output_folder_location}/sensitivity_predictions_volatility.csv",
        index=False,
    )
    sensitivity_evaluation_df.to_csv(
        f"{output_folder_location}/sensitivity_output_evaluation.csv",
        index=False,
    )

    # STEP 4: Run tool interaction
    RunInteraction(csv_path, output, output_folder_location, verbose).main(df)
