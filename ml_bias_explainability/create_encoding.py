import tensorflow as tf
from tensorflow.keras import layers


class CreateEncoding:
    """
    Needs the dataset to create encoding, along with a list of numeric and object columns.

    Returns a the encoded dataset and model input format.
    """

    def main(self, numeric_list, object_list, ds):
        all_inputs = []
        encoded_features = []

        # Numeric features.
        for header in numeric_list:
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            normalization_layer = self._get_normalization_layer(header, ds)
            encoded_numeric_col = normalization_layer(numeric_col)
            all_inputs.append(numeric_col)
            encoded_features.append(encoded_numeric_col)

        # Categorical features encoded as string.
        for header in object_list:
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype="string")
            encoding_layer = self._get_category_encoding_layer(
                header, ds, dtype="string", max_tokens=5
            )
            encoded_categorical_col = encoding_layer(categorical_col)
            all_inputs.append(categorical_col)
            encoded_features.append(encoded_categorical_col)

        return encoded_features, all_inputs

    def _get_normalization_layer(self, name, dataset):
        # Create a Normalization layer for our feature.
        normalizer = layers.Normalization(axis=None)

        # Prepare a Dataset that only yields our feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)

        return normalizer

    def _get_category_encoding_layer(self, name, dataset, dtype, max_tokens=None):
        # Create a layer that turns strings into integer indices.
        if dtype == "string":
            index = layers.StringLookup(max_tokens=max_tokens)
        # Otherwise, create a layer that turns integer values into integer indices.
        else:
            index = layers.IntegerLookup(max_tokens=max_tokens)

        # Prepare a `tf.data.Dataset` that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)

        # Encode the integer indices.
        encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

        # Apply multi-hot encoding to the indices. The lambda function captures the
        # layer, so you can use them, or include them in the Keras Functional model later.
        return lambda feature: encoder(index(feature))
