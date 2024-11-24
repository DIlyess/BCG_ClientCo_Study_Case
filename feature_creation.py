import pandas as pd
import numpy as np


class FeatureCreation:

    def __init__(self, df_yield, df_climate) -> None:
        self.df_climate = df_climate
        self.df_yield = df_yield
        self.df_hist = df_yield
        self.df_forecast = pd.DataFrame()
        self.group_col = ["scenario", "nom_dep", "code_dep", "year"]
        self.metrics = df_climate.columns[~df_climate.columns.isin(self.group_col)]

    def add_amplitude(self, frequency):
        # Define the aggregation function
        def amplitude(group):
            return group.max() - group.min()

        # Group data per department, scenario and resample it to the desired frequency
        monthly_data = (
            self.df_climate.groupby(by=self.group_col)
            .resample(frequency)
            .agg(amplitude)
            .reset_index()
        )
        # Rename columns and format date from 'YYYY-MM-DD' to 'MMM'
        new_features = ["amp_" + feat for feat in self.metrics]
        col_rename = {
            old_feat: new_feat for old_feat, new_feat in zip(self.metrics, new_features)
        }
        monthly_data.rename(columns=col_rename, inplace=True)
        monthly_data["date"] = monthly_data["date"].apply(
            lambda date: date.strftime("%b")
        )

        # Pivot from monthly data to yearly data
        pivoted_df = monthly_data.pivot_table(
            index=["scenario", "nom_dep", "code_dep", "year"],
            columns="date",
            values=new_features,
        ).reset_index()
        pivoted_df.columns = [
            f"{metric}_{month}".strip("_") for metric, month in pivoted_df.columns
        ]
        pivoted_df.set_index(["year"], drop=True, inplace=True)
        pivoted_df.index = pd.to_datetime(pivoted_df.index, format="%Y")

        # Add the yearly calculated feature to the historical df and the forecast df
        self.df_hist = pd.merge(
            self.df_hist,
            pivoted_df[pivoted_df["scenario"] == "historical"],
            left_on=["department", "year"],
            right_on=["nom_dep", "year"],
            how="outer",
        )

        # Initialisation of the forecast df
        self.df_forecast = pivoted_df[~(pivoted_df["scenario"] == "historical")]
