import numpy as np
from keras import layers, models
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from ml_bias_explainability.helpers import Helper


class FindBestParameter:
    """
    Needs the output feature and a range of parameters for the grid search.

    Returns a dict of the best values for each parameter to use in the DL Model.
    """

    def __init__(self, output, param_grid, cv=3, n_jobs=-1, verbose=2):
        self.output = output
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def main(self, df):
        grid = self._perform_grid_search_cv(Helper.convert_types(df.copy()))
        return grid.best_params_

    def _perform_grid_search_cv(self, df):
        # Create x, y for grid search
        x_train = df.copy()

        output_raw = np.array(x_train.pop(self.output))
        label_encoder = LabelEncoder()
        output_encoded = label_encoder.fit_transform(output_raw)
        y_train = to_categorical(output_encoded)  # defining as 2D output tensor

        # Create NN model
        input_size = len(x_train.iloc[0])
        self.output_size = len(y_train[0])
        optimizer_model = KerasClassifier(
            build_fn=self._create_grid_model, input_size=input_size, verbose=0
        )

        # Train models using grid search params
        grid_model = GridSearchCV(
            estimator=optimizer_model,
            param_grid=self.param_grid,
            n_jobs=self.n_jobs,
            cv=self.cv,
            verbose=self.verbose,
        )
        grid = grid_model.fit(x_train, y_train)

        # Output best results
        self._summarize_hyperparameter_tuning_results(grid)

        return grid

    def _create_grid_model(
        self,
        input_size=None,
        optimizer=None,
        init_mode=None,
        activation=None,
        weight_constraint=None,
        dropout_rate=None,
        neurons=None,
        hidden_layers=None,
    ):
        if self.output_size == 2:
            activation = "sigmoid"
            loss = "binary_crossentropy"
        else:
            activation = "softmax"
            loss = "categorical_crossentropy"

        # create model
        model = models.Sequential()
        for i in range(hidden_layers):
            model.add(
                layers.Dense(
                    neurons,
                    input_dim=input_size,
                    kernel_initializer=init_mode,
                    activation=activation,
                    kernel_constraint=maxnorm(weight_constraint),
                )
            )
        model.add(layers.Dropout(dropout_rate))
        model.add(
            layers.Dense(self.output_size, kernel_initializer=init_mode, activation=activation)
        )
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        return model

    def _summarize_hyperparameter_tuning_results(self, grid):
        print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
