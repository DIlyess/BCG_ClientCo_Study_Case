import pandas as pd
import numpy as np


class ETL:
    def __init__(self, df_yield, df_climate) -> None:
        self.df_yield = df_yield
        self.df_climate = df_climate

    def structure_df_climate(self):
        """
        This method
            - pivots the df_climate DataFrame to have the metric as columns
            - sets the index to the time column
            - converts the index to a datetime object
        """
        df_climate = self.df_climate
        df_climate = df_climate.pivot(
            index=["time", "scenario", "nom_dep", "code_dep", "year"],
            columns="metric",
            values="value",
        ).reset_index()
        df_climate.columns.name = None

        df_climate.set_index("time", inplace=True, drop=True)
        df_climate.index = pd.to_datetime(df_climate.index)
        df_climate.index = df_climate.index.date
        df_climate.index.name = "date"
        self.df_climate = df_climate

    def clean_df_climate(self):
        # Temporary : drop all NaNs
        df_climate = self.df_climate
        missing_data = df_climate[df_climate.isna().any(axis=1)]
        missing_data = missing_data[["nom_dep", "scenario"]].drop_duplicates(
            subset=["nom_dep", "scenario"]
        )
        print(
            f"Departments/Scenario dropped because of any missing values: {missing_data}"
        )
        df_climate.dropna(inplace=True)
        self.df_climate = df_climate

    def structure_df_yield(self):
        """
        This method
            - sets the index to the time column
            - converts the index to a datetime object
        """
        df_yield = self.df_yield
        df_yield.drop(columns=["Unnamed: 0"], inplace=True)
        df_yield = df_yield.set_index("year", drop=True)
        df_yield.index = pd.to_datetime(df_yield.index, format="%Y")
        self.df_yield = df_yield

    def clean_df_yield(self):
        """
        This methods removes the departments with more than 70% of missing values for ['yield', 'production', 'area']
        """
        df_yield = self.df_yield
        missing_yield = df_yield[df_yield.isna().any(axis=1)]
        missing_per_dep = missing_yield.groupby("department").agg(self.custom_agg)

        dep_to_drop = missing_per_dep[
            missing_per_dep.mean(axis=1) > len(df_yield.index.unique()) * 0.7
        ].index
        df_yield = df_yield[~df_yield["department"].isin(dep_to_drop)]
        print(
            f"Departments dropped because of almost absolute absence of data:\n {dep_to_drop.values}"
        )

        # Set 0 values for production and area to NaN
        df_yield.loc[df_yield["area"] == 0, "area"] = np.nan
        df_yield.loc[df_yield["production"] == 0, "production"] = np.nan

        self.df_yield = df_yield

    def impute_df_yield(self):
        """
        This method imputes the missing values in the df_yield DataFrame
        """
        df_yield = self.df_yield
        # Case 1: yield is missing but production and area are not
        mask = (
            df_yield["yield"].isna()
            & df_yield["production"].notna()
            & df_yield["area"].notna()
            & df_yield["area"]
            != 0
        )
        df_yield.loc[mask, "yield"] = df_yield.loc[mask, "production"].div(
            df_yield.loc[mask, "area"]
        )

        # Case 2: yield is missing + area and/or production missing -> interpolate
        mask = (
            (df_yield["yield"].isna())
            & (df_yield["production"].isna())
            & (df_yield["area"].isna())
        )
        for dep in df_yield[mask]["department"].unique():
            df_dep = df_yield[df_yield["department"] == dep]
            for col in ["production", "area", "yield"]:
                df_dep.loc[:, col] = df_dep.loc[:, col].interpolate()
            df_dep.loc[:, "yield"] = (
                df_dep.loc[:, "yield"]
                + df_dep.loc[:, "production"] / df_dep.loc[:, "area"]
            ) / 2
            df_yield.loc[df_yield["department"] == dep] = df_dep

        # Case 3: yield is present but production and/or area are missing
        # We do not care as the yield will be our target.

        self.df_yield = df_yield

    def run(self):
        self.structure_df_climate()
        self.clean_df_climate()
        self.structure_df_yield()
        self.clean_df_yield()
        self.impute_df_yield()
        return self.df_yield, self.df_climate

    @staticmethod
    def custom_agg(row):
        return row.isna().sum()
