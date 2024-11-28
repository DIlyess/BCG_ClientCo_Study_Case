import pandas as pd
import numpy as np


class FeatureCreation:
    def __init__(self, df_yield, df_climate) -> None:
        self.df_climate = df_climate
        self.df_yield = df_yield
        self.df_hist = df_yield
        self.df_forecast = None
        self.df_phase = None
        self.group_col = ["scenario", "department", "code_dep", "year"]
        self.metrics = df_climate.columns[~df_climate.columns.isin(self.group_col)]

    def run(self):
        # Call feature creation methods in the correct order
        self.add_stat_features()
        self.add_threshold_days()
        self.add_temperature_categories()
        self.add_consecutive_days()
        self.add_annual_precipitation_category()
        self.add_yield_lagged_features()
        self.add_estimate_future_yields()
        # self.add_department_coordinates()

        self.df_hist = self.df_hist.reset_index()
        self.df_hist["year"] = self.df_hist["year"].apply(lambda x: x.year)

        self.df_forecast = self.df_forecast.reset_index()
        self.df_forecast["year"] = self.df_forecast["year"].apply(lambda x: x.year)

        # WARNING TMP --- Removing departments ---
        # self.df_hist = self.encode_departments(self.df_hist)
        # self.df_forecast = self.encode_departments(self.df_forecast)

        return self.df_hist, self.df_forecast

    def add_stat_features(self):
        # Map dates to growth phases
        """
        def map_date_to_phase(date):
            month = date.month
            if month in [10, 11]:  # October, November
                return 'germination'
            elif month in [12, 1, 2, 3, 4, 5]:  # December to May
                return 'growth'
            elif month in [6, 7]:  # June, July
                return 'maturation'
            else:
                return None
        """

        # Map dates to months
        def map_date_to_month(date):
            return date.month

        def map_date_to_quarter(date):
            month = date.month
            if month in [1, 2, 3]:
                return "Q1"
            elif month in [4, 5, 6]:
                return "Q2"
            elif month in [7, 8, 9]:
                return "Q3"
            else:
                return "Q4"

        # Reset index to ensure 'date' is a column
        df_climate = self.df_climate.reset_index()

        # Map dates to months
        df_climate["phase"] = df_climate["date"].apply(map_date_to_quarter)

        # Update grouping columns to include 'phase'
        group_col = self.group_col + ["phase"]

        # Define amplitude calculation
        def amplitude(group):
            return group.max() - group.min()

        def mean(group):
            return group.mean()

        def std(group):
            return group.std()

        dic_agg = {metric: [amplitude, mean, std] for metric in self.metrics}

        # Group by the new grouping columns and calculate amplitude
        phase_data = df_climate.groupby(by=group_col).agg(dic_agg).reset_index()

        # Rename columns
        rename_dic = {(col, "amplitude"): f"{col}_amplitude" for col in self.metrics}
        rename_dic.update({(col, "mean"): f"{col}_mean" for col in self.metrics})
        rename_dic.update({(col, "std"): f"{col}_std" for col in self.metrics})
        phase_data.columns = [
            (
                "_".join(filter(None, [str(c) for c in col])).strip("_")
                if isinstance(col, tuple)
                else col
            )
            for col in phase_data.columns.values
        ]

        # Pivot the data to have phases as columns
        pivoted_df = phase_data.pivot_table(
            index=[
                "scenario",
                "department",
                "code_dep",
                "year",
            ],  # Ensure 'year' is included
            columns="phase",
            values=list(rename_dic.values()),
        ).reset_index()  # Keep 'year' and other grouping columns

        # Flatten the MultiIndex columns
        pivoted_df.columns = [
            (
                "_".join(filter(None, [str(c) for c in col])).strip("_")
                if isinstance(col, tuple)
                else col
            )
            for col in pivoted_df.columns.values
        ]

        # Set 'year' as the index
        pivoted_df.set_index("year", inplace=True)
        pivoted_df.index = pd.to_datetime(pivoted_df.index, format="%Y")

        # Merge with historical data
        self.df_hist = pd.merge(
            self.df_hist,
            pivoted_df[pivoted_df["scenario"] == "historical"],
            left_on=["department", "year"],
            right_on=["department", "year"],
            how="inner",
        )

        # Initialize the forecast dataframe
        self.df_forecast = pivoted_df[~(pivoted_df["scenario"] == "historical")]

        print("--- Amplitude feature created over phases ---")

    def add_threshold_days(self):
        thresholds = {"rain": 8.493e-05, "frost": 0 + 273.15, "heat": 30 + 273.15}

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

        # Reset index to ensure 'date' is a column
        df_climate = self.df_climate.reset_index()

        # Map dates to phases
        def map_date_to_phase(date):
            month = date.month
            if month in [10, 11]:  # October, November
                return "germination"
            elif month in [12, 1, 2, 3, 4, 5]:  # December to May
                return "growth"
            elif month in [6, 7]:  # June, July
                return "maturation"
            else:
                return None

        df_climate["phase"] = df_climate["date"].apply(map_date_to_phase)
        df_climate = df_climate[df_climate["phase"].notnull()]  # Filter valid phases

        group_col = self.group_col + ["phase"]

        # Apply aggregation
        phase_data = df_climate.groupby(by=group_col).agg(dic_agg).reset_index()

        # Rename columns
        rename_dic = {col: func.__name__ for col, func in dic_agg.items()}
        phase_data.rename(columns=rename_dic, inplace=True)

        # Pivot the data to have phases as columns
        pivoted_df = phase_data.pivot_table(
            index=[
                "scenario",
                "department",
                "code_dep",
                "year",
            ],  # Ensure 'year' is included
            columns="phase",
            values=list(rename_dic.values()),
        ).reset_index()

        # Flatten the MultiIndex columns
        pivoted_df.columns = [
            "_".join(filter(None, col)).strip("_") if isinstance(col, tuple) else col
            for col in pivoted_df.columns.values
        ]

        # Set 'year' as the index
        pivoted_df.set_index("year", inplace=True)
        pivoted_df.index = pd.to_datetime(pivoted_df.index, format="%Y")

        # Merge with historical and forecast data
        self.df_hist = pd.merge(
            self.df_hist,
            pivoted_df[pivoted_df["scenario"] == "historical"],
            on=["department", "year", "code_dep", "scenario"],
            how="inner",
        )

        self.df_forecast = pd.merge(
            self.df_forecast,
            pivoted_df[~(pivoted_df["scenario"] == "historical")],
            on=["department", "year", "scenario", "code_dep"],
            how="inner",
        )

        print("--- Threshold days feature created over phases ---")

    def add_temperature_categories(self):
        # Define phases and temperature ranges
        phases = {
            "germination": {
                "temp_range": (20 + 273.15, 25 + 273.15),
                "months": [10, 11],
            },
            "growth": {
                "temp_range": (5 + 273.15, 22 + 273.15),
                "months": [12, 1, 2, 3, 4, 5],
            },
            "maturation": {"temp_range": (15 + 273.15, 25 + 273.15), "months": [6, 7]},
        }

        # Reset index to ensure 'date' is a column
        df_climate = self.df_climate.reset_index()

        # Map months to corresponding phases
        month_to_phase = {
            month: phase for phase, info in phases.items() for month in info["months"]
        }
        df_climate["phase"] = df_climate["date"].dt.month.map(month_to_phase)

        # Filter rows that belong to defined phases
        df_phase = df_climate[df_climate["phase"].notnull()].copy()

        # Map temperature ranges to the respective phases
        df_phase["temp_min"] = df_phase["phase"].map(
            lambda x: phases[x]["temp_range"][0]
        )
        df_phase["temp_max"] = df_phase["phase"].map(
            lambda x: phases[x]["temp_range"][1]
        )

        # Categorize temperatures
        def temp_category(row):
            if row["daily_NSA_temp"] < row["temp_min"]:
                return "below"
            elif row["daily_NSA_temp"] > row["temp_max"]:
                return "above"
            else:
                return "within"

        df_phase["temp_category"] = df_phase.apply(temp_category, axis=1)

        # Count the number of days in each temperature category for each phase
        counts = (
            df_phase.groupby(
                ["scenario", "department", "code_dep", "year", "phase", "temp_category"]
            )
            .size()
            .reset_index(name="days_count")
        )

        # Pivot the data to create a wide format with separate columns for each category and phase
        counts_pivot = counts.pivot_table(
            index=["scenario", "department", "code_dep", "year"],
            columns=["phase", "temp_category"],
            values="days_count",
            fill_value=0,
        ).reset_index()

        # Flatten the MultiIndex columns
        counts_pivot.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in counts_pivot.columns.values
        ]

        # Set 'year' as the index
        counts_pivot.set_index("year", inplace=True)
        counts_pivot.index = pd.to_datetime(counts_pivot.index, format="%Y")

        # Split data into historical and forecast datasets
        counts_pivot_hist = counts_pivot[counts_pivot["scenario"] == "historical"]
        counts_pivot_forecast = counts_pivot[counts_pivot["scenario"] != "historical"]

        # Merge the new features into the historical dataset
        self.df_hist = pd.merge(
            self.df_hist,
            counts_pivot_hist,
            on=["scenario", "department", "code_dep", "year"],
            how="inner",
        )

        # Merge the new features into the forecast dataset
        self.df_forecast = pd.merge(
            self.df_forecast,
            counts_pivot_forecast,
            on=["scenario", "department", "code_dep", "year"],
            how="inner",
        )

        # **Assign df_phase to self.df_phase**
        self.df_phase = df_phase

        print("--- Temperature categories feature created ---")

    def add_consecutive_days(self):
        df_phase = self.df_phase.copy()

        df_phase["is_below"] = df_phase["temp_category"] == "below"
        df_phase["is_above"] = df_phase["temp_category"] == "above"
        df_phase["is_within"] = df_phase["temp_category"] == "within"

        df_phase.sort_values(
            by=["scenario", "department", "code_dep", "year", "phase", "date"],
            inplace=True,
        )

        def compute_total_consecutive_days(group):
            total_below = self.total_consecutive_days(group["is_below"])
            total_above = self.total_consecutive_days(group["is_above"])
            return pd.Series(
                {
                    "max_consecutive_below": total_below,
                    "max_consecutive_above": total_above,
                }
            )

        total_consecutive_days_df = (
            df_phase.groupby(["scenario", "department", "code_dep", "year", "phase"])
            .apply(compute_total_consecutive_days)
            .reset_index()
        )

        total_consecutive_days_pivot = total_consecutive_days_df.pivot_table(
            index=["scenario", "department", "code_dep", "year"],
            columns="phase",
            values=[
                "max_consecutive_below",
                "max_consecutive_above",
            ],
        )

        total_consecutive_days_pivot.columns = [
            "_".join([metric, phase])
            for metric, phase in total_consecutive_days_pivot.columns
        ]

        total_consecutive_days_pivot = total_consecutive_days_pivot.reset_index()

        # Set 'year' as the index
        total_consecutive_days_pivot.set_index("year", inplace=True)
        total_consecutive_days_pivot.index = pd.to_datetime(
            total_consecutive_days_pivot.index, format="%Y"
        )

        total_consecutive_days_hist = total_consecutive_days_pivot[
            total_consecutive_days_pivot["scenario"] == "historical"
        ]
        total_consecutive_days_forecast = total_consecutive_days_pivot[
            total_consecutive_days_pivot["scenario"] != "historical"
        ]

        self.df_hist = pd.merge(
            self.df_hist,
            total_consecutive_days_hist,
            on=["scenario", "department", "code_dep", "year"],
            how="inner",
        )

        self.df_forecast = pd.merge(
            self.df_forecast,
            total_consecutive_days_forecast,
            on=["scenario", "department", "code_dep", "year"],
            how="inner",
        )

        print("--- Consecutive days feature created ---")

    def total_consecutive_days(self, series):
        max_count = 0
        count = 0
        for val in series:
            if val:
                count += 1
                if count > max_count:
                    max_count = count
            else:
                count = 0
        return max_count

    def add_annual_precipitation_category(self):
        # Reset index to ensure 'date' is a column
        df_climate = self.df_climate.reset_index()
        # Sum the precipitation per year, per scenario, per department, per code_dep
        annual_precipitation = (
            df_climate.groupby(["scenario", "department", "code_dep", "year"])[
                "precipitation"
            ]
            .sum()
            .reset_index()
        )

        # Classify the annual precipitation
        def classify_precipitation(amount):
            if amount < 450 / 86400:
                return {
                    "annual_precip_below": 1,
                    "annual_precip_within": 0,
                    "annual_precip_above": 0,
                }
            elif 450 / 86400 <= amount <= 650 / 86400:
                return {
                    "annual_precip_below": 0,
                    "annual_precip_within": 1,
                    "annual_precip_above": 0,
                }
            else:
                return {
                    "annual_precip_below": 0,
                    "annual_precip_within": 0,
                    "annual_precip_above": 1,
                }

        # Apply the classification and create new columns
        precip_categories = annual_precipitation["precipitation"].apply(
            classify_precipitation
        )
        precip_categories_df = pd.DataFrame(list(precip_categories))
        annual_precipitation = pd.concat(
            [annual_precipitation, precip_categories_df], axis=1
        )
        annual_precipitation = annual_precipitation.reset_index()

        # Set 'year' as the index
        annual_precipitation.set_index("year", inplace=True)
        annual_precipitation.index = pd.to_datetime(
            annual_precipitation.index, format="%Y"
        )

        # Now merge this back into df_hist and df_forecast
        # Separate historical and forecast data
        hist_data = annual_precipitation[
            annual_precipitation["scenario"] == "historical"
        ]
        forecast_data = annual_precipitation[
            annual_precipitation["scenario"] != "historical"
        ]

        # Merge into df_hist
        self.df_hist = pd.merge(
            self.df_hist,
            hist_data[
                [
                    "scenario",
                    "department",
                    "code_dep",
                    "annual_precip_below",
                    "annual_precip_within",
                    "annual_precip_above",
                ]
            ],
            on=["scenario", "department", "code_dep", "year"],
            how="inner",
        )

        # Merge into df_forecast
        self.df_forecast = pd.merge(
            self.df_forecast,
            forecast_data[
                [
                    "scenario",
                    "department",
                    "code_dep",
                    "annual_precip_below",
                    "annual_precip_within",
                    "annual_precip_above",
                ]
            ],
            on=["scenario", "department", "code_dep", "year"],
            how="inner",
        )

        print("--- Annual precipitation category feature created ---")

    def add_yield_lagged_features(self):
        print("--- Calculating lagged features for yield ---")

        # Sort data to ensure proper lag calculation
        self.df_hist.sort_values(by=["department", "year"], inplace=True)

        # Create lagged features for yield
        # self.df_hist['yield_lag1'] = self.df_hist.groupby('department')['yield'].shift(1)  # Lag of 1 year
        self.df_hist["cagr"] = self.df_hist.groupby("department")[
            "yield"
        ].pct_change()  # Percentage change

        print("--- Yield lagged features calculated ---")

    def add_estimate_future_yields(self):
        print("--- Estimating future yields using historical CAGR ---")
        # Reset index to ensure 'year' is a column
        df_hist_yields = self.df_hist.reset_index()
        df_future_yields = self.df_forecast.reset_index()

        # Convert 'year' to datetime and extract the year
        df_hist_yields["year"] = pd.to_datetime(df_hist_yields["year"], format="%Y")
        df_hist_yields["year"] = df_hist_yields["year"].dt.year
        df_future_yields["year"] = pd.to_datetime(df_future_yields["year"], format="%Y")
        df_future_yields["year"] = df_future_yields["year"].dt.year

        # Ensure 'year' is an integer
        df_hist_yields["year"] = df_hist_yields["year"].astype(int)
        df_future_yields["year"] = df_future_yields["year"].astype(int)

        # Compute CAGR per department
        def compute_cagr(group):
            first_year = group["year"].min()
            last_year = group["year"].max()
            num_years = last_year - first_year
            beginning_value = group.loc[group["year"] == first_year, "yield"].values[0]
            ending_value = group.loc[group["year"] == last_year, "yield"].values[0]
            if num_years > 0 and beginning_value > 0:
                cagr = (ending_value / beginning_value) ** (1 / num_years) - 1
            else:
                cagr = 0
            return cagr

        cagr_df = df_hist_yields.groupby("department").apply(compute_cagr).reset_index()
        cagr_df.rename(columns={0: "cagr"}, inplace=True)

        # Get the last historical yield per department
        last_yield_df = (
            df_hist_yields.sort_values(by=["department", "year"])
            .groupby("department")
            .tail(1)[["department", "year", "yield"]]
        )

        # Reset index to ensure 'year' is a column in df_forecast
        self.df_forecast = self.df_forecast.reset_index()

        # Merge the CAGR into the forecast data
        self.df_forecast = pd.merge(
            self.df_forecast,
            cagr_df[["department", "cagr"]],
            on="department",
            how="inner",
        )
        # Check if 'year' is still present
        if "year" in self.df_forecast.columns:
            # If needed, set 'year' back as the index
            self.df_forecast.set_index("year", inplace=True)
        else:
            print("Warning: 'year' column is missing in df_forecast after merging.")

        print("--- Future yields CAGR ---")

    def add_department_coordinates(self):
        # Load department coordinates from a local CSV file
        department_coords = pd.read_csv("departments_coordinates.csv")

        # Ensure 'year' is a column before merging
        self.df_hist = (
            self.df_hist.reset_index()
            if "year" not in self.df_hist.columns
            else self.df_hist.copy()
        )
        self.df_forecast = (
            self.df_forecast.reset_index()
            if "year" not in self.df_forecast.columns
            else self.df_forecast.copy()
        )

        # Merge coordinates into the historical DataFrame
        self.df_hist = pd.merge(
            self.df_hist,
            department_coords[["department", "latitude", "longitude"]],
            on="department",
            how="inner",
        )

        # Restore 'year' as the index if needed
        if "year" in self.df_hist.columns:
            self.df_hist.set_index("year", inplace=True)
        else:
            print("Warning: 'year' column is missing in df_hist after the merge.")

        # Merge coordinates into the forecast DataFrame
        self.df_forecast = pd.merge(
            self.df_forecast,
            department_coords[["department", "latitude", "longitude"]],
            on="department",
            how="inner",
        )

        # Drop rows where we added NaNs (ie when the department was not in the df_hist or df_forecast)

        # Restore 'year' as the index if needed
        if "year" in self.df_forecast.columns:
            self.df_forecast.set_index("year", inplace=True)
        else:
            print("Warning: 'year' column is missing in df_forecast after the merge.")

        print("--- Department coordinates added ---")

    @staticmethod
    def encode_departments(df):
        dep_col = df.columns[df.columns.str.contains("dep|department")]
        # enc = OneHotEncoder()
        # enc.fit(df[[dep_col]])
        # dep_encoded = enc.transform(df[[dep_col]]).toarray()
        # df = pd.concat(
        #     [df, pd.DataFrame(dep_encoded, columns=enc.get_feature_names([dep_col]))],
        #     axis=1,
        # ).drop(columns=[dep_col])

        # For now we just remove the department column
        df = df.drop(columns=[dep_col])
        print("Dropped :", dep_col)
        return df
