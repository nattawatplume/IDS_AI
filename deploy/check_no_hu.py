"""Build guard: the image must never carry Hu's private labels (file or DB rows)."""

import sys
from pathlib import Path

import duckdb


def main(data_dir: Path) -> int:
    data_dir = Path(data_dir)
    bad: list[str] = []
    hu_file = data_dir / "labels" / "hu_pd_logs.tsv"
    if hu_file.exists():
        bad.append(f"พบไฟล์ {hu_file}")
    for db in sorted(data_dir.rglob("*.duckdb")):
        con = duckdb.connect(str(db), read_only=True)
        try:
            has = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'labels'"
            ).fetchone()[0]
            if has:
                n = con.execute("SELECT count(*) FROM labels WHERE source = 'hu'").fetchone()[0]
                if n:
                    bad.append(f"พบแถวป้ายกำกับของ hu จำนวน {n} แถวใน {db}")
        finally:
            con.close()
    for msg in bad:
        print(f"ข้อผิดพลาด: {msg} - ห้ามนำข้อมูลส่วนตัวของ Hu ลงในอิมเมจ", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "data")))
