import pandas as pd
import oracledb
from pathlib import Path

# =========================
# CONFIG
# =========================
ORACLE_USER = "username"
ORACLE_PASSWORD = "password"
ORACLE_DSN = "host:1521/service_name"
# Example: "dbserver.company.com:1521/ORCLPDB1"

QUERY = """
SELECT *
FROM schema_name.table_name
WHERE created_date >= :start_date
"""

PARAMS = {
    "start_date": "2025-01-01"
}

CHUNKSIZE = 100_000      # None for small tables
OUTPUT_DIR = "./output"
OUTPUT_NAME = "oracle_data"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# CONNECTION
# =========================
def get_connection():
    return oracledb.connect(
        user=ORACLE_USER,
        password=ORACLE_PASSWORD,
        dsn=ORACLE_DSN
    )

# =========================
# READ + PROCESS
# =========================
def read_oracle(query, params=None, chunksize=None):
    conn = get_connection()
    try:
        return pd.read_sql(
            query,
            conn,
            params=params,
            chunksize=chunksize
        )
    finally:
        conn.close()

# =========================
# MAIN
# =========================
if CHUNKSIZE is None:
    # ---- Small / medium dataset ----
    df = read_oracle(QUERY, PARAMS)

    print(df.head())
    print(df.dtypes)

    df.to_parquet(f"{OUTPUT_DIR}/{OUTPUT_NAME}.parquet", index=False)
    df.to_csv(f"{OUTPUT_DIR}/{OUTPUT_NAME}.csv", index=False)

else:
    # ---- Large dataset (chunked) ----
    chunks = read_oracle(QUERY, PARAMS, CHUNKSIZE)

    for i, chunk in enumerate(chunks):
        chunk.to_parquet(
            f"{OUTPUT_DIR}/{OUTPUT_NAME}_part_{i}.parquet",
            index=False
        )
        print(f"Processed chunk {i}, rows={len(chunk)}")

print("✅ Done")
