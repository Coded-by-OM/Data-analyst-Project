# project24_analysis.py
# Project 24: Mobile App Usage & User Engagement Analysis
#
# HOW TO RUN: put this file and 24_mobile_app_usage.csv in the same folder,
# then run:   python project24_analysis.py
#
# It prints every result on screen AND saves files in the same folder:
#   24_cleaned_app_usage.csv, 24_outliers.csv, 24_report.txt

import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# SETTINGS: if your CSV uses different column names, change them here
# ---------------------------------------------------------------
FILE_NAME = "24_mobile_app_usage.csv"
COL_SESSION = "Session_ID"
COL_USER = "User_ID"
COL_DATE = "Session_Date"
COL_FEATURE = "Feature_Used"
COL_DEVICE = "Device_Type"
COL_AGE = "Age_Group"
COL_DURATION = "Session_Duration_Mins"
COL_RATING = "User_Rating"

report_lines = []   # everything we print is also stored here for the report file


def show(text=""):
    """Print on screen and keep a copy for the report file."""
    print(text)
    report_lines.append(str(text))


def heading(title):
    show()
    show("=" * 70)
    show(title)
    show("=" * 70)


# ---------------------------------------------------------------
# TASK 1: Load and inspect
# ---------------------------------------------------------------
heading("TASK 1: Load and inspect the data")
try:
    df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    sys.exit(f"ERROR: Could not find '{FILE_NAME}'. Put it in the same folder as this script.")

needed = [COL_SESSION, COL_USER, COL_DATE, COL_FEATURE, COL_DEVICE,
          COL_AGE, COL_DURATION, COL_RATING]
missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    sys.exit(f"ERROR: These columns are not in your file: {missing_cols}\n"
             f"Your file has: {list(df.columns)}\n"
             f"Fix the names in the SETTINGS section at the top of this script.")

show(f"Rows, columns: {df.shape}")
show(f"Column names: {list(df.columns)}")
show("Data types:")
show(df.dtypes)
show("First 5 rows:")
show(df.head())

# ---------------------------------------------------------------
# TASK 2: Missing values and duplicates
# ---------------------------------------------------------------
heading("TASK 2: Missing values and duplicate Session_IDs")
show("Missing values per column:")
show(df.isnull().sum())
dup_count = df.duplicated(subset=COL_SESSION).sum()
show(f"Duplicate Session_ID rows: {dup_count}")
df = df.drop_duplicates(subset=COL_SESSION, keep="first")
show(f"Rows after removing duplicates: {len(df)}")

# ---------------------------------------------------------------
# TASK 3: Handle missing values
# ---------------------------------------------------------------
heading("TASK 3: Handle missing Session_Duration_Mins and User_Rating")
show("Method: fill missing values with the MEDIAN of each column.")
show("Reason: session times are skewed by a few very long sessions, and the")
show("median is not pulled up by them the way the mean is. Ratings are whole")
show("numbers, so the median keeps a sensible rating value.")
df[COL_DURATION] = pd.to_numeric(df[COL_DURATION], errors="coerce")
df[COL_RATING] = pd.to_numeric(df[COL_RATING], errors="coerce")
df[COL_DURATION] = df[COL_DURATION].fillna(df[COL_DURATION].median())
df[COL_RATING] = df[COL_RATING].fillna(df[COL_RATING].median())
show("Missing values after cleaning:")
show(df[[COL_DURATION, COL_RATING]].isnull().sum())

# ---------------------------------------------------------------
# TASK 4: Dates
# ---------------------------------------------------------------
heading("TASK 4: Convert Session_Date and extract month, weekday, hour")
df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
df = df.dropna(subset=[COL_DATE])     # remove rows whose date could not be read
df["Month"] = df[COL_DATE].dt.month_name()
df["Month_Number"] = df[COL_DATE].dt.month
df["Weekday"] = df[COL_DATE].dt.day_name()
df["Hour"] = df[COL_DATE].dt.hour
if df["Hour"].nunique() == 1:
    show("NOTE: your dates have no time of day, so every Hour is the same.")
    show("Peak-hour results (Task 10) will not be meaningful for this file.")
show(df[[COL_DATE, "Month", "Weekday", "Hour"]].head())

# ---------------------------------------------------------------
# TASK 5: Basic metrics
# ---------------------------------------------------------------
heading("TASK 5: Total sessions, unique users, average session duration")
show(f"Total sessions: {df[COL_SESSION].nunique()}")
show(f"Unique users: {df[COL_USER].nunique()}")
show(f"Average session duration: {df[COL_DURATION].mean():.2f} minutes")

# ---------------------------------------------------------------
# TASK 6: Top 10 features
# ---------------------------------------------------------------
heading("TASK 6: 10 most frequently used features")
top_features = df[COL_FEATURE].value_counts().head(10)
show(top_features)

# ---------------------------------------------------------------
# TASK 7: Duration by device
# ---------------------------------------------------------------
heading("TASK 7: Average session duration by device type")
device_duration = df.groupby(COL_DEVICE)[COL_DURATION].mean().sort_values(ascending=False).round(2)
show(device_duration)

# ---------------------------------------------------------------
# TASK 8: Age groups by sessions
# ---------------------------------------------------------------
heading("TASK 8: Age groups ranked by number of sessions")
age_sessions = df.groupby(COL_AGE)[COL_SESSION].count().sort_values(ascending=False)
show(age_sessions)

# ---------------------------------------------------------------
# TASK 9: Ratings by feature
# ---------------------------------------------------------------
heading("TASK 9: Average user rating by feature")
feature_rating = df.groupby(COL_FEATURE)[COL_RATING].mean().sort_values(ascending=False).round(2)
show(feature_rating)

# ---------------------------------------------------------------
# TASK 10: Peak hours and busiest weekdays
# ---------------------------------------------------------------
heading("TASK 10: Peak usage hours and busiest weekdays")
hours = df["Hour"].value_counts().head(5)
show("Top 5 busiest hours (hour of day : sessions):")
show(hours)
weekdays = df["Weekday"].value_counts()
show("Sessions by weekday:")
show(weekdays)

# ---------------------------------------------------------------
# TASK 11: List, tuple, set
# ---------------------------------------------------------------
heading("TASK 11: List, tuple and set")
most_used_features = top_features.index.tolist()            # a LIST
device_categories = tuple(sorted(df[COL_DEVICE].unique()))  # a TUPLE
unique_users_set = set(df[COL_USER])                        # a SET
show(f"List of most-used features: {most_used_features}")
show(f"Tuple of device categories: {device_categories}")
show(f"Number of unique users (set): {len(unique_users_set)}")

# ---------------------------------------------------------------
# TASK 12: Function
# ---------------------------------------------------------------
heading("TASK 12: Function - average session duration for a feature")


def avg_duration_for_feature(data, feature):
    """Return the average session duration for one feature (or None)."""
    subset = data[data[COL_FEATURE] == feature]
    if subset.empty:
        return None
    return subset[COL_DURATION].mean()


for f in most_used_features[:3]:
    show(f"{f}: {avg_duration_for_feature(df, f):.2f} minutes")

# ---------------------------------------------------------------
# TASK 13: NumPy statistics
# ---------------------------------------------------------------
heading("TASK 13: NumPy statistics for session duration")
d = df[COL_DURATION].to_numpy()
show(f"Mean:   {np.mean(d):.2f}")
show(f"Median: {np.median(d):.2f}")
show(f"Std dev: {np.std(d):.2f}")

# ---------------------------------------------------------------
# TASK 14: IQR outliers and engagement labels
# ---------------------------------------------------------------
heading("TASK 14: Outliers (IQR method) and engagement labels")
q1, q3 = np.percentile(d, [25, 75])
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
show(f"Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
show(f"Normal range: {lower:.2f} to {upper:.2f} minutes")
outliers = df[(df[COL_DURATION] < lower) | (df[COL_DURATION] > upper)]
show(f"Unusually long/short sessions found: {len(outliers)}")
show(outliers[[COL_SESSION, COL_USER, COL_FEATURE, COL_DURATION]].head(10))

# Engagement rules: based on the 33rd and 66th percentiles of duration
p33, p66 = np.percentile(d, [33, 66])
show(f"Engagement rules: Low if <= {p33:.2f} mins, "
     f"Medium if <= {p66:.2f} mins, High if above that.")


def engagement(mins):
    """Label a session as Low, Medium or High engagement."""
    if mins <= p33:
        return "Low"
    elif mins <= p66:
        return "Medium"
    return "High"


df["Engagement"] = df[COL_DURATION].apply(engagement)
show(df["Engagement"].value_counts())

# ---------------------------------------------------------------
# TASK 15: Class and recommendations
# ---------------------------------------------------------------
heading("TASK 15: AppUser class and recommendations")


class AppUser:
    """A simple app user with an ID and a device type."""

    def __init__(self, user_id, device_type):
        self.user_id = user_id
        self.device_type = device_type

    def __repr__(self):
        return f"AppUser(user_id={self.user_id}, device_type={self.device_type})"


first = df.iloc[0]
show(AppUser(first[COL_USER], first[COL_DEVICE]))

# Recommendations are built from YOUR results, so the numbers are real
top_feature = top_features.index[0]
low_rated_feature = feature_rating.index[-1]
best_device = device_duration.index[0]
worst_device = device_duration.index[-1]
top_age = age_sessions.index[0]
peak_hour = int(hours.index[0])
busiest_day = weekdays.index[0]
quietest_day = weekdays.index[-1]
high_share = (df["Engagement"] == "High").mean() * 100

show("Five recommendations:")
show(f"1. '{top_feature}' is the most used feature ({top_features.iloc[0]} sessions). "
     f"Keep it fast and prominent on the home screen.")
show(f"2. '{low_rated_feature}' has the lowest average rating "
     f"({feature_rating.iloc[-1]}). Review its usability and fix pain points.")
show(f"3. {worst_device} users have the shortest sessions "
     f"({device_duration.iloc[-1]} mins vs {device_duration.iloc[0]} on {best_device}). "
     f"Check performance and layout on {worst_device}.")
show(f"4. The {top_age} age group has the most sessions "
     f"({age_sessions.iloc[0]}). Aim new features and campaigns at them.")
if df["Hour"].nunique() > 1:
    show(f"5. Peak usage is around {peak_hour}:00 and {busiest_day} is the busiest day. "
         f"Send notifications just before these times, and run re-engagement offers "
         f"on {quietest_day}, the quietest day.")
else:
    show(f"5. {busiest_day} is the busiest day and {quietest_day} the quietest. "
         f"Run re-engagement offers on {quietest_day}.")
show(f"(Only {high_share:.0f}% of sessions are High engagement, so aim to lift this.)")

# ---------------------------------------------------------------
# SAVE OUTPUT FILES
# ---------------------------------------------------------------
df.to_csv("24_cleaned_app_usage.csv", index=False)
outliers.to_csv("24_outliers.csv", index=False)
with open("24_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\nDONE. Files saved in this folder:")
print("  24_cleaned_app_usage.csv, 24_outliers.csv, 24_report.txt")
