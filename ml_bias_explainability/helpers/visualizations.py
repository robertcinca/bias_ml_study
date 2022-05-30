import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.ioff()


class Visualizations:
    def __init__(self, output_folder_location=None, show=False, save_fig=False, figure_title=None):
        self.show = show
        self.output_folder_location = output_folder_location
        self.save_fig = save_fig
        self.figure_title = figure_title

    def small_multiples_plot(
        self,
        grouped_df,
        delta_columns_of_interest,
        y_label,
        feature,
        feature_value=None,
        background_color=None,
    ):
        df_for_plot = pd.melt(
            grouped_df,
            id_vars=["training_type"],
            value_vars=delta_columns_of_interest,
            ignore_index=False,
        )

        g = sns.FacetGrid(df_for_plot, col="variable", hue="variable", size=5)
        g = g.map(sns.barplot, "training_type", "value")

        g.fig.suptitle(self.figure_title)

        for axes in g.axes.flat:
            axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

            if (
                background_color
            ):  # TODO: see if need to use a different condition var, when new plots are added
                self._set_background_color(axes)

            if axes.get_ylabel():
                axes.set_ylabel(y_label)

            # Highlight all_features label
            for label in axes.xaxis.get_ticklabels():
                if label.get_text() == "all_features":
                    label.set_color("red")

        plt.tight_layout()

        if feature_value:
            plot_name = f"{feature}_{feature_value}"
        else:
            plot_name = feature

        if self.save_fig:
            plt.savefig(f"{self.output_folder_location}/{plot_name}_small_multiples.png")

        if self.show:
            plt.show()

    def stacked_visualization(
        self,
        grouped_df,
        columns_of_interest,
        y_label,
        feature,
        feature_value=None,
        background_color=None,
    ):
        # Plotting of deltas by training type
        grouped_df["sum_delta"] = grouped_df[columns_of_interest].sum(axis=1)

        g = grouped_df.sort_values("sum_delta", ascending=False)[
            columns_of_interest + ["training_type"]
        ].plot(x="training_type", kind="bar", stacked=True, figsize=(8, 8))

        plt.title(self.figure_title)
        plt.xlabel("training_type")
        plt.ylabel(y_label)
        plt.subplots_adjust(bottom=0.31)

        # Highlight all_features label
        for label in g.xaxis.get_ticklabels():
            if label.get_text() == "all_features":
                label.set_color("red")

        if (
            background_color
        ):  # TODO: see if need to use a different condition var, when new plots are added
            self._set_background_color(g)

        if feature_value:
            plot_name = f"{feature}_{feature_value}"
        else:
            plot_name = feature

        if self.save_fig:
            plt.savefig(f"{self.output_folder_location}/{plot_name}_stacked_visualization.png")

        if self.show:
            plt.show()

    def correlation_plot(self, df: pd.DataFrame, correlation_plot_threshold: float = 0.5) -> None:
        """
        Performs a Pearson's correlation coefficient calculation on all
        features. Selects those correlations over a defined threshold
        and plots them on a heatmap.
        """
        df_correlations = df.corr()

        df_filtered = df_correlations[
            (
                (df_correlations >= correlation_plot_threshold)
                | (df_correlations <= -correlation_plot_threshold)
            )
            & (df_correlations != 1.000)
        ].dropna(axis=1, how="all")

        if df_filtered.empty:
            print("Cannot plot correlation heatmap, filtered DataFrame is empty.")
            return

        plt.figure(figsize=(12, 10))
        sns.heatmap(df_filtered, annot=True, cmap="Reds")

        if self.save_fig:
            plt.savefig(
                f"{self.output_folder_location}/visualizations/analysis_1/correlation_plot.png"
            )

        if self.show:
            plt.show()

    def _set_background_color(self, axes):
        y_max = max(axes.get_ylim())
        axes.autoscale(enable=True, axis="y", tight=True)
        axes.axhspan(0, y_max, facecolor="green", alpha=0.2)

        y_min = min(axes.get_ylim())
        axes.autoscale(enable=True, axis="y", tight=True)
        axes.axhspan(0, y_min, facecolor="red", alpha=0.2)
