"""Small artificial input tables; these are not the paper datasets."""

import numpy as np
import pandas as pd


def customer_table():
    i = np.arange(240)
    group = i // 80
    return pd.DataFrame(
        {
            "age": 20 + i % 60 + 0.5,
            "income": 1000 + i * 20.75,
            "gender": np.array(["Male", "Female", "Male"])[group],
            "education": np.array(["HighSchool", "Bachelor", "Masters"])[group],
            "region": np.array(["North", "South", "East"])[group],
            "loyalty_status": np.array(["Regular", "Silver", "Gold"])[group],
            "purchase_frequency": np.array(["rare", "occasional", "frequent"])[group],
            "purchase_amount": 10 + i * 0.8 + group * 10 + np.sin(i),
        }
    )


def seoulbike_table():
    i = np.arange(240)
    group = i // 80
    frame = pd.DataFrame(
        {
            column: i * (j + 1) / 17 + 0.125
            for j, column in enumerate(
                [
                    "Hour",
                    "Temperature",
                    "Humidity",
                    "Wind_speed",
                    "Visibility",
                    "Dew_point_temperature",
                    "Solar_Radiation",
                    "Rainfall",
                    "Snowfall",
                ]
            )
        }
    )
    frame["Seasons"] = np.array(["Spring", "Summer", "Winter"])[group]
    frame["Holiday"] = np.array(["Holiday", "No Holiday", "No Holiday"])[group]
    frame["Functioning_Day"] = np.array(["Yes", "Yes", "No"])[group]
    frame["Rented_Bike_Count"] = 100 + i * 0.9 + group * 10 + np.cos(i)
    return frame


def small_customer_table():
    """Six distinct feature rows keep even the MIQP within a restricted license."""
    frame = customer_table()
    group = np.arange(len(frame)) // 40
    frame["age"] = 20.5 + group * 10
    frame["income"] = 1000.25 + group * 500
    return frame


def data_snapshot(data):
    """Stable digest per view, retaining indices, column order and values."""
    import hashlib
    from dataclasses import asdict

    result = {}
    for name, value in asdict(data).items():
        if isinstance(value, pd.DataFrame):
            text = value.to_json(orient="split", double_precision=12, force_ascii=False)
        elif isinstance(value, np.ndarray):
            text = pd.DataFrame(value).to_json(orient="split", double_precision=12)
        else:
            text = repr(value)
        result[name] = hashlib.sha256(text.encode()).hexdigest()
    return result
