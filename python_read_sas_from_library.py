import pyreadstat
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
SAS_LIBRARY_PATH = "/data/sas/mylib"   # physical path behind the SAS library
OUTPUT_DIR = "./output"
ENCODING = "latin1"                   # common for SAS

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# READ ALL SAS DATASETS
# =========================
for sas_file in Path(SAS_LIBRARY_PATH).glob("*.sas7bdat"):
    print(f"Reading {sas_file.name}...")

    # Read SAS dataset
    df, meta = pyreadstat.read_sas7bdat(
        sas_file,
        encoding=ENCODING
    )

    # Optional: inspect metadata
    print("Columns:", df.columns.tolist())
    print("Labels :", meta.column_labels)

    # =========================
    # SAVE FOR ANALYTICS
    # =========================
    df.to_parquet(
        Path(OUTPUT_DIR) / f"{sas_file.stem}.parquet",
        index=False
    )

    df.to_csv(
        Path(OUTPUT_DIR) / f"{sas_file.stem}.csv",
        index=False
    )

    print(f"✔ Converted {sas_file.stem}")

print("✅ All SAS library datasets processed")
