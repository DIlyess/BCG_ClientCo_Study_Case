def longest_consecutive_years(group):
    """
    mask = (
        (df_yield["yield"].isna())
        & (df_yield["production"].isna())
        & (df_yield["area"].isna() | (df_yield["area"] == 0))
    )
    longest_gap = df_yield[mask].groupby("department").apply(longest_consecutive_years)
    """
    years = group.index.year  # Extract the years from the index
    longest_streak = 0
    current_streak = 1

    for i in range(1, len(years)):
        if years[i] == years[i - 1] + 1:
            current_streak += 1
        else:
            longest_streak = max(longest_streak, current_streak)
            current_streak = 1

    longest_streak = max(longest_streak, current_streak)
    return longest_streak
