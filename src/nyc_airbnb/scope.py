"""Semantic scope rule independent of price levels or model errors."""
import pandas as pd


def primary_mask(frame: pd.DataFrame) -> pd.Series:
    hospitality = frame["room_type"].eq("Hotel room") | frame["property_type"].fillna("").str.contains(
        "hotel|hostel|resort", case=False, regex=True)
    return ~hospitality
