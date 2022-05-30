import tensorflow as tf
from keras.constraints import maxnorm


class GenerateModel:
    """
    Needs the output feature, the best parameters for the model and the encoded datasets.

    Returns a compiled model.
    """

    def __init__(self, best_params, output_size):
        self.best_params = best_params
        self.output_size = output_size

    def main(self, encoded_features, all_inputs, train_ds, val_ds, test_ds):
        all_features = tf.keras.layers.concatenate(encoded_features)

        # Initial layer
        x = tf.keras.layers.Dense(
            self.best_params["neurons"],
            kernel_initializer=self.best_params["init_mode"],
            kernel_constraint=maxnorm(self.best_params["weight_constraint"]),
            activation=self.best_params["activation"],
        )(all_features)

        # Inner layers
        for i in range(self.best_params["hidden_layers"] - 1):
            x = tf.keras.layers.Dense(
                self.best_params["neurons"],
                kernel_initializer=self.best_params["init_mode"],
                kernel_constraint=maxnorm(self.best_params["weight_constraint"]),
                activation=self.best_params["activation"],
            )(x)

        # Dropout rate
        x = tf.keras.layers.Dropout(self.best_params["dropout_rate"])(x)

        # Output
        if self.output_size == 1:
            activation = "sigmoid"
            loss = "binary_crossentropy"
        else:
            activation = "softmax"
            loss = "categorical_crossentropy"

        output = tf.keras.layers.Dense(self.output_size, activation)(x)

        # Model processing
        model = tf.keras.Model(all_inputs, output)

        model.compile(
            optimizer=self.best_params["optimizer"],
            loss=loss,
            metrics=[
                tf.keras.metrics.FalseNegatives(),
                tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.TruePositives(),
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        print("Training the model:")
        model.fit(
            train_ds,
            epochs=self.best_params["epochs"],
            validation_data=val_ds,
        )

        return model
