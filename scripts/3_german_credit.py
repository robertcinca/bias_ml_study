import bootstrap  # noqa
import pandas as pd

from ml_bias_explainability.compile_model_and_analysis import CompileModelAndAnalysis
from ml_bias_explainability.helpers.helper import Helper
from ml_bias_explainability.run_tool_interaction import RunInteraction

# from ml_bias_explainability.find_best_parameters import FindBestParameter


def clean_dataset(csv_path, columns_to_remove):
    # The dataset used is taken from https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    df = pd.read_csv(csv_path, index_col=0)

    # get rid of columns that shouldn't be in the input model
    filtered_columns_to_remove = df.filter(columns_to_remove)
    df.drop(filtered_columns_to_remove, inplace=True, axis=1)

    # lowercase + underscore for columns
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")

    # map output to integer values
    df["risk"].replace({"bad": 0, "good": 1}, inplace=True)

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

    features_of_interest = ["age", "sex"]
    output = "risk"
    columns_to_remove = Helper.read_yaml("scripts/data/3_german_credit/columns_to_remove.yml")
    csv_path = "scripts/data/3_german_credit/german_credit_data.csv"
    output_folder_location = "scripts/data/3_german_credit/output"
    unique_values = 10
    verbose = False

    # Param grid: either a yaml file with multiple options or hard coded
    # param_grid = Helper.read_yaml("scripts/data/3_german_credit/param_grid.yml")
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
        batch_size=256,
        epochs=15,
        optimizer="Adam",
        init_mode="he_uniform",
        activation="relu",
        weight_constraint=3,
        dropout_rate=0.1,
        neurons=64,
        hidden_layers=2,
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
