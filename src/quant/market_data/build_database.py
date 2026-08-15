from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import duckdb


def sql_path(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def extract_archives(root: Path, extracted: Path) -> None:
    extracted.mkdir(parents=True, exist_ok=True)
    for archive_path in sorted(root.glob("[0-9][0-9][0-9][0-9].zip")):
        year = archive_path.stem
        year_dir = extracted / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        print(f"[extract] {archive_path.name}", flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                if entry.is_dir() or not entry.filename.lower().endswith(".parquet"):
                    continue
                filename = Path(entry.filename).name
                target = year_dir / filename
                if target.exists() and target.stat().st_size == entry.file_size:
                    continue
                temporary = target.with_suffix(".parquet.part")
                with archive.open(entry) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
                temporary.replace(target)


def normalize_months(extracted: Path, normalized: Path) -> None:
    normalized.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET preserve_insertion_order = false")
    try:
        for year_dir in sorted(extracted.glob("year=[0-9][0-9][0-9][0-9]")):
            year = int(year_dir.name.split("=", 1)[1])
            source = sql_path(year_dir / "*.parquet")
            months = con.execute(
                f"SELECT DISTINCT CAST(substr(date, 5, 2) AS INTEGER) AS month "
                f"FROM read_parquet('{source}', union_by_name=true) ORDER BY month"
            ).fetchall()
            for (month,) in months:
                target_dir = normalized / f"year={year}" / f"month={month:02d}"
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / "bars.parquet"
                if target.exists() and target.stat().st_size > 0:
                    print(f"[skip] {year}-{month:02d}", flush=True)
                    continue
                temporary = target_dir / "bars.parquet.part"
                print(f"[normalize] {year}-{month:02d}", flush=True)
                query = f"""
                    SELECT
                        CAST(code AS VARCHAR) AS code,
                        CAST(trade_time AS TIMESTAMP) AS trade_time,
                        CAST(strptime(date, '%Y%m%d') AS DATE) AS trade_date,
                        CAST(open AS DOUBLE) AS open,
                        CAST(high AS DOUBLE) AS high,
                        CAST(low AS DOUBLE) AS low,
                        CAST(close AS DOUBLE) AS close,
                        CAST(round(vol) AS BIGINT) AS volume,
                        CAST(amount AS DOUBLE) AS amount,
                        CAST(pre_close AS DOUBLE) AS pre_close,
                        CAST(change AS DOUBLE) AS change,
                        CAST(pct_chg AS DOUBLE) AS pct_change
                    FROM read_parquet('{source}', union_by_name=true)
                    WHERE substr(date, 5, 2) = '{month:02d}'
                    ORDER BY trade_time, code
                """
                con.execute(
                    f"COPY ({query}) TO '{sql_path(temporary)}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3, "
                    "ROW_GROUP_SIZE 500000)"
                )
                temporary.replace(target)
    finally:
        con.close()


def create_catalog(root: Path, normalized: Path) -> Path:
    database = root / "market.duckdb"
    parquet_glob = sql_path(normalized / "year=*" / "month=*" / "*.parquet")
    con = duckdb.connect(str(database))
    try:
        con.execute("DROP VIEW IF EXISTS bars_5m")
        con.execute(
            f"""
            CREATE VIEW bars_5m AS
            SELECT *
            FROM read_parquet(
                '{parquet_glob}',
                hive_partitioning=true,
                union_by_name=true
            )
            """
        )
        con.execute("CREATE TABLE IF NOT EXISTS dataset_metadata(key VARCHAR PRIMARY KEY, value VARCHAR)")
        metadata = {
            "dataset": "A-share 5-minute bars",
            "schema_version": "1",
            "parquet_glob": parquet_glob,
            "source_root": sql_path(root),
        }
        for key, value in metadata.items():
            con.execute("INSERT OR REPLACE INTO dataset_metadata VALUES (?, ?)", [key, value])

        con.execute("DROP TABLE IF EXISTS monthly_inventory")
        con.execute(
            """
            CREATE TABLE monthly_inventory AS
            SELECT year, month, min(trade_time) AS min_time, max(trade_time) AS max_time,
                   count(*) AS row_count, count(DISTINCT code) AS symbol_count,
                   count(DISTINCT trade_date) AS trading_days
            FROM bars_5m GROUP BY year, month ORDER BY year, month
            """
        )
        con.execute("DROP TABLE IF EXISTS symbols")
        con.execute(
            """
            CREATE TABLE symbols AS
            SELECT code, min(trade_date) AS first_date, max(trade_date) AS last_date,
                   count(*) AS bar_count
            FROM bars_5m GROUP BY code ORDER BY code
            """
        )
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return database


def main() -> None:
    parser = argparse.ArgumentParser(description="构建5分钟行情Parquet数据集和DuckDB目录")
    parser.add_argument("--root", type=Path, default=Path(r"D:\量化"))
    args = parser.parse_args()
    root = args.root.resolve()
    extracted = root / "extracted_daily"
    normalized = root / "bars_5m"
    extract_archives(root, extracted)
    normalize_months(extracted, normalized)
    database = create_catalog(root, normalized)
    print(f"[done] database={database}")


if __name__ == "__main__":
    main()
