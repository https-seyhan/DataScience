import pandas as pd
import pyreadstat
from pathlib import Path

# =========================
# CONFIG
# =========================
SAS_FILE = "/path/to/file.sas7bdat"
OUTPUT_DIR = "./output"
ENCODING = "latin1"      # try 'utf-8' if this fails
CHUNKSIZE = None         # set to e.g. 100_000 for large files

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# OPTION 1: pyreadstat (recommended)
# =========================
def read_sas_with_metadata(path):
    df, meta = pyreadstat.read_sas7bdat(path)

    print("Columns:", df.columns.tolist())
    print("Column labels:", meta.column_labels)

    return df, meta


# =========================
# OPTION 2: pandas (supports chunking)
# =========================
def read_sas_pandas(path, chunksize=None):
    return pd.read_sas(
        path,
        format="sas7bdat",
        encoding=ENCODING,
        chunksize=chunksize
    )


# =========================
# SAS DATE CONVERSION
# =========================
def convert_sas_dates(df):
    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            # Heuristic: SAS dates are usually between 0 and ~50,000
            if df[col].between(0, 50000, inclusive="both").any():
                try:
                    df[col] = pd.to_datetime(
                        df[col],
                        unit="D",
                        origin="1960-01-01",
                        errors="ignore"
                    )
                except Exception:
                    pass
    return df


# =========================
# MAIN
# =========================
if CHUNKSIZE is None:
    # ---- Small / medium dataset ----
    df, meta = read_sas_with_metadata(SAS_FILE)
    df = convert_sas_dates(df)

    # Save outputs
    df.to_parquet(f"{OUTPUT_DIR}/data.parquet", index=False)
    df.to_csv(f"{OUTPUT_DIR}/data.csv", index=False)

else:
    # ---- Large dataset (chunked) ----
    chunks = read_sas_pandas(SAS_FILE, chunksize=CHUNKSIZE)

    for i, chunk in enumerate(chunks):
        chunk = convert_sas_dates(chunk)
        chunk.to_parquet(
            f"{OUTPUT_DIR}/data_part_{i}.parquet",
            index=False
        )
        print(f"Processed chunk {i}, rows={len(chunk)}")

print("✅ Done")
