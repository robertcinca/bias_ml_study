import bootstrap  # noqa
import pandas as pd

from ml_bias_explainability.compile_model_and_analysis import CompileModelAndAnalysis
from ml_bias_explainability.helpers.helper import Helper
from ml_bias_explainability.run_tool_interaction import RunInteraction


def clean_dataset(csv_path, columns_to_remove):
    # The dataset used is taken from https://openpolicing.stanford.edu/data/
    df = pd.read_csv(csv_path)

    # Get rid of rows where columns that matter have NaN
    df.dropna(subset=["outcome"], inplace=True)

    # get rid of columns that shouldn't be in the input model
    filtered_columns_to_remove = df.filter(columns_to_remove)
    df.drop(filtered_columns_to_remove, inplace=True, axis=1)

    # Map bool to 0 and 1
    df.replace({False: 0, True: 1}, inplace=True)

    # map output to integer values
    df["outcome"].replace({"citation": 0, "warning": 1, "arrest": 2}, inplace=True)

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

    features_of_interest = ["subject_race", "subject_sex", "subject_age"]
    output = "outcome"
    columns_to_remove = Helper.read_yaml("scripts/data/2_stop_and_search/columns_to_remove.yml")

    csv_path = "scripts/data/2_stop_and_search/ca_san_diego_2020_04_01.csv"
    # csv_path = "scripts/data/2_stop_and_search/ct_hartford_2020_04_01.csv"
    # csv_path = "scripts/data/2_stop_and_search/vt_burlington_2020_04_01.csv"
    output_folder_location = "scripts/data/2_stop_and_search/output"
    unique_values = 10
    verbose = False

    # Param grid: either a yaml file with multiple options or hard coded
    # param_grid = Helper.read_yaml("scripts/data/2_stop_and_search/param_grid.yml")
    param_grid = dict(
        batch_size=[128, 256],
        epochs=[15],
        optimizer=["RMSprop", "Adam"],
        init_mode=["he_uniform", "glorot_uniform"],
        activation=["relu"],
        weight_constraint=[1, 2, 3],
        dropout_rate=[0.01, 0.1],
        neurons=[32, 64],
        hidden_layers=[1, 2, 3],
    )

    # STEP 1: Get cleaned dataset
    df = clean_dataset(csv_path, columns_to_remove)

    # STEP 2 [OPT]: Get best parameters (either compile at run time or define)
    # best_params = FindBestParameter(output, param_grid).main(df)
    best_params = dict(
        batch_size=128,
        epochs=15,
        optimizer="Adam",
        init_mode="he_uniform",
        activation="relu",
        weight_constraint=0,
        dropout_rate=0.001,
        neurons=16,
        hidden_layers=1,
    )

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

    model_evaluation_df.round(2).to_csv(
        f"{output_folder_location}/ml_models_evaluation.csv",
        index=False,
    )
    raw_predictions_df.round(2).to_csv(
        f"{output_folder_location}/sensitivity_predictions_raw.csv",
        index=False,
    )
    predictions_change_df.round(2).to_csv(
        f"{output_folder_location}/sensitivity_predictions_change.csv",
        index=False,
    )
    volatility_df.round(2).to_csv(
        f"{output_folder_location}/sensitivity_predictions_volatility.csv",
        index=False,
    )
    sensitivity_evaluation_df.round(2).to_csv(
        f"{output_folder_location}/sensitivity_output_evaluation.csv",
        index=False,
    )

    # STEP 4: Run tool interaction
    RunInteraction(csv_path, output, output_folder_location, verbose).main(df)
