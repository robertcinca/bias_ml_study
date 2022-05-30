import json
import re
import statistics as s

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split


class Helper:
    @staticmethod
    def read_json(file_path):
        with open(file_path) as stream:
            return json.load(stream)

    @staticmethod
    def read_yaml(file_path):
        with open(file_path) as stream:
            return yaml.safe_load(stream)

    @staticmethod
    def convert_types(df):
        """
        Convert df to tensor-accepted types. Numeric followed by categorical columns
        """
        # Replace bool with numeric
        df.replace({False: 0.0, True: 1.0}, inplace=True)

        object_list, numeric_list = Helper.get_df_types_list(df)

        # convert numeric types
        df[numeric_list] = df[numeric_list].astype(np.float32)

        # convert objects
        for column in object_list:
            df[column] = pd.Categorical(df[column])
            df[column] = df[column].cat.codes
        return df

    @staticmethod
    def get_df_types_list(df):
        """
        Gets a list of columns that are numeric and objects
        """
        numeric_list = df.select_dtypes(include=[np.number]).columns
        object_list = df.select_dtypes(object).columns
        return object_list, numeric_list

    @staticmethod
    def get_input_column_dtypes(dataframe):
        """
        Gets a list of columns that are numeric and objects
        """
        numeric_list = list(dataframe.select_dtypes(include=[np.number]).columns)
        object_list = list(dataframe.select_dtypes(object).columns)

        if "target" in numeric_list:
            numeric_list.remove("target")
        elif "target" in object_list:
            object_list.remove("target")

        print(f"List of numeric features for this dataset: {numeric_list}")
        print(f"List of non-numeric features: {object_list}\n")

        return numeric_list, object_list

    @staticmethod
    def define_output(dataframe, output):
        # Define target
        dataframe["target"] = dataframe[output]

        # Drop un-used columns.
        dataframe = dataframe.drop(columns=[output])

        # name new output
        output = "target"

        return dataframe, output

    @staticmethod
    def train_test_split(dataframe):
        train, test = train_test_split(dataframe, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)

        print(f"Total number of training samples: {len(train)}")
        print(f"Total number of validation samples: {len(val)}")
        print(f"Total number of test samples: {len(test)}\n")

        return train, test, val

    @staticmethod
    def df_to_dataset(dataframe):
        """
        A utility method to create a tf.data dataset from a Pandas Dataframe
        """
        df = dataframe.copy()

        # for 1D output tensor
        labels = df.pop("target")
        output_size = 1

        # In future version: for 2D output tensor
        # output_raw = np.array(labels)
        # label_encoder = LabelEncoder()
        # output_encoded = label_encoder.fit_transform(output_raw)
        # labels = to_categorical(output_encoded)
        # output_size = len(labels[0])

        # raise Exception()
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

        return ds, output_size

    def split_ds_into_sets(df_ds, batch_size):
        def is_test(x, y):  # 20% test set
            return x % 5 == 0

        def is_train_val(x, y):
            return not is_test(x, y)

        def is_val(x, y):  # 20% val set
            return x % 4 == 0

        def is_train(x, y):
            return not is_val(x, y)

        test_ds = df_ds.enumerate().filter(is_test).map(lambda x, y: y)
        train_val_ds = df_ds.enumerate().filter(is_train_val).map(lambda x, y: y)

        val_ds = train_val_ds.enumerate().filter(is_val).map(lambda x, y: y)
        train_ds = train_val_ds.enumerate().filter(is_train).map(lambda x, y: y)

        train_ds = train_ds.batch(batch_size)
        val_ds = val_ds.batch(batch_size)
        test_ds = test_ds.batch(batch_size)

        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds

    @staticmethod
    def extract_df_from_ds(ds):
        """
        Extract df from tf dataset

        input: ds - an unbatched tf dataset
        output: df - a dataframe containing input + output features
        """
        features_dict, labels_dict = [], []

        for element in ds.as_numpy_iterator():
            element_list = list(element)

            features_dict.append(element_list[0])
            labels_dict.append(element_list[1])

        output_df = pd.DataFrame.from_dict(labels_dict)
        output_df.columns = ["target"]

        input_df = pd.DataFrame.from_dict(features_dict)

        # convert byte column values to string
        for col, dtype in input_df.dtypes.items():

            if dtype == np.object:  # Only process object columns.
                # decode, or return original value if decode return Nan
                input_df[col] = input_df[col].str.decode("utf-8").fillna(input_df[col])

        # merge input and output dfs into one and return
        return pd.merge(input_df, output_df, left_index=True, right_index=True)

    @staticmethod
    def remove_certain_features(df, feature_to_remove):
        return df.drop(columns=[feature_to_remove], errors="ignore")

    @staticmethod
    def equalized_odds(
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    ):
        """
        Fairness Definition: Equalized Odds
        """
        return s.harmonic_mean(
            [
                (true_positives)
                / (true_positives + false_positives + true_negatives + false_negatives),
                (false_positives)
                / (true_positives + false_positives + true_negatives + false_negatives),
            ]
        )

    @staticmethod
    def equal_opportunity(
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    ):
        """
        Fairness Definition: Equal Opportunity
        """
        return (true_positives) / (
            true_positives + false_positives + true_negatives + false_negatives
        )

    @staticmethod
    def statistical_parity(
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    ):
        """
        Fairness Definition: Statistical Parity
        """
        return (true_positives + false_positives) / (
            true_positives + false_positives + true_negatives + false_negatives
        )

    @staticmethod
    def treatment_equality(
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
    ):
        """
        Fairness Definition: Treatment Equality
        """
        false_positive_rate = (false_positives) / (
            true_positives + false_positives + true_negatives + false_negatives
        )
        false_negative_rate = (false_negatives) / (
            true_positives + false_positives + true_negatives + false_negatives
        )
        return false_positive_rate / (1 + false_negative_rate)

    @staticmethod
    def strip_text_for_file_naming(text):
        # remove any special characters
        text = re.sub(r"\W", " ", text)

        # Substituting multiple spaces with single space
        text = re.sub(r"\s+", " ", text, flags=re.I)

        # Converting to Lowercase
        text = text.lower()

        # Remove leading and ending spaces
        text = text.strip()

        # Replace space with _ to meet formatting requirements of file naming
        text = text.replace(" ", "_")

        return text
