import pyreadstat
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
SAS_LIBRARY_PATH = "/data/sas/mylib"   # physical path backing the SAS library
DATASET_NAME = "customers"            # SAS dataset name (no extension)
ENCODING = "latin1"                   # common SAS encoding
OUTPUT_DIR = "./output"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# BUILD FILE PATH
# =========================
sas_file = Path(SAS_LIBRARY_PATH) / f"{DATASET_NAME}.sas7bdat"

if not sas_file.exists():
    raise FileNotFoundError(f"SAS dataset not found: {sas_file}")

# =========================
# READ SAS DATASET
# =========================
df, meta = pyreadstat.read_sas7bdat(
    sas_file,
    encoding=ENCODING
)

# =========================
# INSPECT (OPTIONAL)
# =========================
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Column labels:", meta.column_labels)

# =========================
# SAVE FOR ANALYTICS
# =========================
df.to_parquet(
    Path(OUTPUT_DIR) / f"{DATASET_NAME}.parquet",
    index=False
)

df.to_csv(
    Path(OUTPUT_DIR) / f"{DATASET_NAME}.csv",
    index=False
)

print("✅ Dataset successfully loaded and saved")
