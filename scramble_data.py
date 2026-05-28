import pandas as pd
import numpy as np
import os
from faker import Faker

fake = Faker()
np.random.seed(42)

# ---- Helper functions ----

def shuffle_column(series):
    return pd.Series(np.random.permutation(series.values), index=series.index)

def randomise_numeric(series):
    if series.nunique() <= 5:
        return shuffle_column(series)

    mean = series.mean()
    std = series.std()

    if std == 0 or np.isnan(std):
        return series

    new_vals = np.random.normal(mean, std, size=len(series))

    if pd.api.types.is_integer_dtype(series):
        new_vals = np.round(new_vals)

    return pd.Series(new_vals).clip(lower=series.min(), upper=series.max())

def fake_names(series):
    return pd.Series([fake.name() for _ in range(len(series))])

def fake_ids(series):
    return pd.Series(np.random.permutation(range(100000, 100000 + len(series))))

def fake_dates(series):
    if not pd.api.types.is_datetime64_any_dtype(series):
        return series

    offset = np.random.randint(-365, 365)
    return series + pd.to_timedelta(offset, unit='D')


def process_dataframe(df):
    df_new = df.copy()

    for col in df.columns:
        col_lower = col.lower()

        # ---- ID columns ----
        if "employee" in col_lower or "id" in col_lower:
            df_new[col] = fake_ids(df[col])

        # ---- Name columns ----
        elif "name" in col_lower:
            df_new[col] = fake_names(df[col])

        # ---- Date columns ----
        elif "date" in col_lower:
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
                df_new[col] = fake_dates(df[col])
            except:
                df_new[col] = df[col]

        # ---- Numeric ----
        elif pd.api.types.is_numeric_dtype(df[col]):
            df_new[col] = randomise_numeric(df[col])

        # ---- Text / categorical ----
        elif pd.api.types.is_object_dtype(df[col]):
            df_new[col] = shuffle_column(df[col])

        else:
            df_new[col] = df[col]

    return df_new


# ---- Main processing ----

data_folder = "."

for file in os.listdir(data_folder):
    if file.endswith(".csv") or file.endswith(".xlsx"):
        print(f"Processing {file}")

        path = os.path.join(data_folder, file)

        try:
            # ---- Load file ----
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(path)
                except:
                    df = pd.read_csv(path, encoding="latin1")

            else:
                df = pd.read_excel(path)

            # ---- Scramble ----
            df_scrambled = process_dataframe(df)

            # ---- Save back ----
            if file.endswith(".csv"):
                df_scrambled.to_csv(path, index=False)
            else:
                df_scrambled.to_excel(path, index=False)

        except Exception as e:
            print(f"Skipping {file}: {e}")

print("All files processed safely")
