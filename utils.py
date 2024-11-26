def total_consecutive_days(condition_series):
    """
    mask = (
        (df_yield["yield"].isna())
        & (df_yield["production"].isna())
        & (df_yield["area"].isna() | (df_yield["area"] == 0))
    )
    longest_gap = df_yield[mask].groupby("department").apply(longest_consecutive_years)
    """

    current_streak = 0
    total_days = 0

    for value in condition_series:
        if value:
            current_streak += 1
        else:
            total_days += current_streak
            current_streak = 0

    total_days += current_streak
    return total_days
