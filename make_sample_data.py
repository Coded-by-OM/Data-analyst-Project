# make_sample_data.py
# Creates a practice dataset called 24_mobile_app_usage.csv
# ONLY run this if you do not have your own dataset file.

import numpy as np
import pandas as pd

np.random.seed(42)          # makes the random data the same every time
n = 1000                    # number of rows

features = ["Home Feed", "Search", "Chat", "Notifications", "Profile",
            "Payments", "Settings", "Video Player", "Wishlist", "Offers"]
feature_weights = [0.22, 0.16, 0.14, 0.10, 0.08, 0.08, 0.05, 0.08, 0.04, 0.05]

devices = ["Android", "iOS", "Tablet"]
age_groups = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]

dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(
    np.random.randint(0, 300 * 24 * 60, n), unit="m")

df = pd.DataFrame({
    "Session_ID": [f"S{i:05d}" for i in range(1, n + 1)],
    "User_ID": [f"U{np.random.randint(1, 301):04d}" for _ in range(n)],
    "Session_Date": dates,
    "Feature_Used": np.random.choice(features, n, p=feature_weights),
    "Device_Type": np.random.choice(devices, n, p=[0.55, 0.35, 0.10]),
    "Age_Group": np.random.choice(age_groups, n, p=[0.07, 0.28, 0.30, 0.17, 0.11, 0.07]),
    "Session_Duration_Mins": np.round(np.random.gamma(2.2, 5.0, n), 1),
    "User_Rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.20, 0.35, 0.30]).astype(float),
})

# Add some messy data on purpose, so the cleaning steps have work to do
df.loc[np.random.choice(n, 40, replace=False), "Session_Duration_Mins"] = np.nan
df.loc[np.random.choice(n, 50, replace=False), "User_Rating"] = np.nan
df = pd.concat([df, df.sample(12, random_state=1)], ignore_index=True)  # duplicates

df.to_csv("24_mobile_app_usage.csv", index=False)
print("Created 24_mobile_app_usage.csv with", len(df), "rows")
