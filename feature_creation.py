import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class FeatureCreation:

    def __init__(self, df_yield, df_climate) -> None:
        self.df_climate = df_climate
        self.df_yield = df_yield
        self.df_hist = df_yield
        self.df_forecast = pd.DataFrame()
        self.group_col = ["scenario", "nom_dep", "code_dep", "year"]
        self.metrics = df_climate.columns[~df_climate.columns.isin(self.group_col)]

    def run(self):
        thresholds = {"rain": 8.493e-05, "frost": 0 + 273.15, "heat": 30 + 273.15}

        # Warning : the order matter as add_amplitude initialize the forecast df
        self.add_amplitude("1M")
        self.add_threshold_days("1M", thresholds)

        self.df_hist = self.df_hist.reset_index()
        self.df_hist["year"] = self.df_hist["year"].apply(lambda x: x.year)

        # WARNING TMP --- Removing departments ---
        self.df_hist = self.encode_departments(self.df_hist)
        self.df_forecast = self.encode_departments(self.df_forecast)

        return self.df_hist, self.df_forecast

    def add_threshold_days(self, frequency, thresholds):

        def rainy_days(group):
            return (group > thresholds["rain"]).sum()

        def frost_days(group):
            return (group < thresholds["frost"]).sum()

        def heat_days(group):
            return (group > thresholds["heat"]).sum()

        rain_col = "precipitation"
        max_temp_col = "max_daily_NSA_temp"
        mean_temp_col = "daily_NSA_temp"

        dic_agg = {
            rain_col: rainy_days,
            max_temp_col: heat_days,
            mean_temp_col: frost_days,
        }

        # To change to resammple per phase
        monthly_data = (
            self.df_climate.groupby(by=self.group_col)
            .resample(frequency)
            .agg(dic_agg)
            .reset_index()
        )

        rename_dic = {col: func.__name__ for col, func in dic_agg.items()}
        monthly_data.rename(columns=rename_dic, inplace=True)

        # To change to rename into ['germination', 'croissance', 'maturation']
        monthly_data["date"] = monthly_data["date"].apply(
            lambda date: date.strftime("%b")
        )

        # Pivot from monthly data to yearly data
        pivoted_df = monthly_data.pivot_table(
            index=["scenario", "nom_dep", "code_dep", "year"],
            columns="date",
            values=list(rename_dic.values()),
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
            left_on=["nom_dep", "year", "code_dep", "scenario"],
            right_on=["nom_dep", "year", "code_dep", "scenario"],
            how="outer",
        )

        # Merge with forecast data
        self.df_forecast = pd.merge(
            self.df_forecast,
            pivoted_df[~(pivoted_df["scenario"] == "historical")],
            on=["nom_dep", "year", "scenario", "code_dep"],
            how="outer",
        )
        for metric in rename_dic.values():
            print(f"--- {metric} feature created ---")

    @staticmethod
    def encode_departments(df):
        dep_col = "department" if "department" in df.columns else "nom_dep"
        # enc = OneHotEncoder()
        # enc.fit(df[[dep_col]])
        # dep_encoded = enc.transform(df[[dep_col]]).toarray()
        # df = pd.concat(
        #     [df, pd.DataFrame(dep_encoded, columns=enc.get_feature_names([dep_col]))],
        #     axis=1,
        # ).drop(columns=[dep_col])

        # For now we just remove the department column
        df = df.drop(columns=[dep_col])
        return df

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
        print("--- Amplitude feature created ---")
