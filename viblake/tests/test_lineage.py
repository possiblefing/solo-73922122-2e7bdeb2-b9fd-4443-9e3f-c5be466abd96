"""数据血缘（lineage）专项测试：每条记录的来源上传文件 ID 与非空数据行序号。

覆盖的需求点：
1. 上传后每条记录都带“原上传文件 ID + 第几条非空数据行（不含表头）”；
   内容完全相同的重复记录各自保留独立来源；
2. 多轮压实后来源仍指向最初的上传批次，不会变成压实合并件；
3. 旧三列 Parquet 仍可读：旧批次件按文件 + 行序恢复来源，
   旧压实件无法判定的行显示“来源未知”（两字段为 null）；
4. 新旧文件混合压实时，未知行保持未知，不会被误记到别的批次；
5. 查询固定受理时快照内的文件，旧件回收不影响来源解析；
6. CSV 中的空行不计入非空数据行序号。
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import parquet_store as ps  # noqa: E402
from app.config import Settings  # noqa: E402
from app.csv_import import parse_readings  # noqa: E402
from app.db import Database  # noqa: E402
from app.gates import PauseGate  # noqa: E402
from app.compaction import CompactionManager  # noqa: E402
from app.service import DataLakeService  # noqa: E402

TMP_ROOT = Path(__file__).resolve().parent / "tmp_lineage"
DAY = "2026-11-13"


def make_settings(case: str) -> Settings:
    root = TMP_ROOT / case
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return Settings(data_dir=str(root), max_batch_rows=10000)


def csv_bytes(rows: list[tuple[str, str, float]], blank_after: int | None = None) -> bytes:
    lines = ["turbine,ts_utc,vibration"]
    for i, (t, ts, v) in enumerate(rows, start=1):
        lines.append(f"{t},{ts},{v}")
        if blank_after is not None and i == blank_after:
            lines.append("")  # 空行：解析时跳过，不占非空数据行序号
    return ("\n".join(lines) + "\n").encode()


def ts(hour: int, minute: int = 0) -> str:
    return f"{DAY}T{hour:02d}:{minute:02d}:00Z"


class Harness:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.db_path)
        self.gate = PauseGate()
        self.cm = CompactionManager(self.db, settings.lake_dir, self.gate)
        self.svc = DataLakeService(self.db, settings, self.gate, self.cm)

    async def start(self) -> None:
        ps.ensure_dirs(self.settings.lake_dir)
        await self.db.connect()
        await self.gate.load(self.db)
        await self.cm.start_reaper()

    async def stop(self) -> None:
        await self.svc.close()
        await self.cm.close()
        await self.db.close()

    async def upload(self, rows, name: str = "x.csv", blank_after: int | None = None):
        return await self.svc.upload_csv(csv_bytes(rows, blank_after), name)

    async def query_all(self, **kw):
        return await self.svc.query(kw.get("turbine"), kw.get("start"), kw.get("end"), 50000)

    async def compact(self, day: str = DAY) -> dict:
        cid, _ = await self.svc.start_compaction(day)
        for _ in range(200):
            row = await self.svc.compaction_status(cid)
            if row["status"] != "running":
                if row["status"] == "failed":
                    raise AssertionError(f"压实失败: {row['error']}")
                return dict(row)
            await asyncio.sleep(0.05)
        raise AssertionError("压实超时未完成")


PASSED = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED.append(f"{name} {detail}")
        print(f"  ✗ {name} {detail}")


async def test_lineage_on_upload() -> None:
    print("[L1] 上传即带来源：文件 ID + 非空数据行序号，重复行各有来源")
    s = make_settings("l1_upload")
    h = Harness(s)
    await h.start()
    try:
        rows = [
            ("WTG-01", ts(8, 0), 1.11),
            ("WTG-01", ts(8, 0), 1.11),  # 与上一行内容完全相同
            ("WTG-02", ts(9, 0), 2.22),
        ]
        up = await h.upload(rows)
        r = await h.query_all()
        check("返回 3 行", r["matched_rows"] == 3)
        for i, row in enumerate(r["rows"], start=1):
            check(
                f"第 {i} 行来源指向本次上传批次 {up['file_id'][:12]}…",
                row["source_file_id"] == up["file_id"],
                str(row),
            )
        check(
            "非空数据行序号为 1/2/3",
            [row["source_row_no"] for row in r["rows"]] == [1, 2, 3],
        )
        # 两条内容相同的记录来源行号不同（仍是不同来源的两条）
        dupes = [x for x in r["rows"] if x["vibration"] == 1.11]
        check(
            "内容相同的两条记录各有独立行序号",
            len(dupes) == 2 and {x["source_row_no"] for x in dupes} == {1, 2},
        )

        # 第二个批次：来源切换到新的上传文件 ID，行序号重新从 1 开始
        up2 = await h.upload([("WTG-03", ts(10), 3.33)])
        r2 = await h.query_all(turbine="WTG-03")
        check(
            "第二批次来源为新文件 ID 且行号从 1 开始",
            r2["rows"][0]["source_file_id"] == up2["file_id"]
            and r2["rows"][0]["source_row_no"] == 1,
        )
        check("两个批次文件 ID 不同", up["file_id"] != up2["file_id"])
    finally:
        await h.stop()


async def test_blank_lines_not_counted() -> None:
    print("[L2] 空行不计入非空数据行序号")
    s = make_settings("l2_blank")
    h = Harness(s)
    await h.start()
    try:
        # 校验层：5 条数据行，第 2 条后插一个空行，data_line_no 仍为 1..5
        readings = parse_readings(
            csv_bytes(
                [("W", ts(8, i), float(i)) for i in range(5)],
                blank_after=2,
            ),
            s,
        )
        check(
            "空行被跳过且序号连续 1..5",
            [x.source_row_no for x in readings] == [1, 2, 3, 4, 5],
            str([x.source_row_no for x in readings]),
        )
        up = await h.upload(
            [("W", ts(8, i), float(i)) for i in range(5)], blank_after=2
        )
        r = await h.query_all()
        check(
            "落盘后行序号仍为 1..5",
            [x["source_row_no"] for x in r["rows"]] == [1, 2, 3, 4, 5],
        )
        check("来源均为同一上传件", {x["source_file_id"] for x in r["rows"]} == {up["file_id"]})
    finally:
        await h.stop()


async def test_lineage_survives_multi_round_compaction() -> None:
    print("[L3] 多轮压实后来源仍是最初上传批次，不变成合并件")
    s = make_settings("l3_rounds")
    h = Harness(s)
    await h.start()
    try:
        up1 = await h.upload([("WTG-01", ts(8, j), float(j)) for j in range(4)])
        up2 = await h.upload([("WTG-02", ts(9, j), float(j + 10)) for j in range(3)])
        await h.compact()  # 第一轮：2 个批次件 -> 1 个压实件

        # 再上传新批次，与压实件共存，随后第二轮压实
        up3 = await h.upload([("WTG-03", ts(10, j), float(j + 20)) for j in range(2)])
        await h.compact()  # 第二轮：压实件 + 新批次件 -> 新压实件

        r = await h.query_all()
        check("两轮压实后总行数 9", r["matched_rows"] == 9, str(r["matched_rows"]))
        check("当前快照只有 1 个文件（二次压实件）", r["files_in_snapshot"] == 1)

        def src_of(turbine: str):
            return {(x["source_file_id"], x["source_row_no"]) for x in r["rows"] if x["turbine"] == turbine}

        check(
            "WTG-01 四行仍指向第一批上传件 up1，行号 1..4",
            src_of("WTG-01") == {(up1["file_id"], i) for i in range(1, 5)},
            str(src_of("WTG-01")),
        )
        check(
            "WTG-02 三行仍指向第二批 up2，行号 1..3",
            src_of("WTG-02") == {(up2["file_id"], i) for i in range(1, 4)},
        )
        check(
            "WTG-03 两行仍指向第三批 up3，行号 1..2",
            src_of("WTG-03") == {(up3["file_id"], i) for i in range(1, 3)},
        )
        # 任何一行的来源都不可以是压实合并件
        compact_ids = {
            j["new_file_id"]
            for j in await h.db.list_compactions()
            if j["status"] == "published"
        }
        used_sources = {x["source_file_id"] for x in r["rows"]}
        check(
            "没有任何行把压实件记为来源",
            not (used_sources & compact_ids),
            f"used={used_sources} compact={compact_ids}",
        )
        check("全部行来源已知（无 null）", all(x["source_file_id"] for x in r["rows"]))
    finally:
        await h.stop()


async def test_legacy_three_column_files() -> None:
    print("[L4] 旧三列 Parquet 兼容：旧批次件恢复来源，旧压实件来源未知")
    s = make_settings("l4_legacy")
    h = Harness(s)
    await h.start()
    try:
        # 手工构造“历史遗留”三列 Parquet，并按现行元数据规则登记
        legacy_batch_id = ps.new_file_id("batch")
        legacy_compact_id = ps.new_file_id("compact")
        legacy_table = pa.table(
            {
                "turbine": pa.array(["WTG-01", "WTG-01", "WTG-02"], type=pa.string()),
                "ts_utc": pa.array(
                    [
                        datetime(2026, 11, 13, 8, 0, 0),
                        datetime(2026, 11, 13, 8, 1, 0),
                        datetime(2026, 11, 13, 9, 0, 0),
                    ],
                    type=pa.timestamp("ns", tz="UTC"),
                ),
                "vibration": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            },
            schema=ps.LEGACY_SCHEMA,
        )
        batch_rel = ps.final_path(DAY, legacy_batch_id)
        compact_rel = ps.final_path(DAY, legacy_compact_id)
        os.makedirs(os.path.join(s.lake_dir, f"day={DAY}"), exist_ok=True)
        pq.write_table(legacy_table, os.path.join(s.lake_dir, batch_rel))
        pq.write_table(legacy_table, os.path.join(s.lake_dir, compact_rel))
        await h.db.insert_file(legacy_batch_id, DAY, batch_rel, 3, ["WTG-01", "WTG-02"])
        await h.db.insert_file(legacy_compact_id, DAY, compact_rel, 3, ["WTG-01", "WTG-02"])

        r = await h.query_all()
        check("旧三列件全部可读，共 6 行", r["matched_rows"] == 6, str(r["matched_rows"]))

        from_batch = [
            x for x in r["rows"] if x["source_file_id"] == legacy_batch_id
        ]
        from_compact = [
            x for x in r["rows"] if x["source_file_id"] is None
        ]
        check(
            "旧批次件 3 行按文件 + 行序恢复来源（行号 1..3）",
            len(from_batch) == 3
            and sorted(x["source_row_no"] for x in from_batch) == [1, 2, 3],
            str(from_batch),
        )
        check(
            "旧压实件 3 行无法判定 → 来源未知（两字段 null）",
            len(from_compact) == 3
            and all(x["source_row_no"] is None for x in from_compact),
            str(from_compact),
        )
        # 未知行绝不能挂到旧批次件名下
        check(
            "未知行没有被误记到旧批次件",
            all(x["source_file_id"] != legacy_batch_id for x in from_compact),
        )

        # 对“旧批次件 + 旧压实件”做一次混合压实，未知状态必须逐行保留
        job = await h.compact()
        check("混合压实发布成功", job["status"] == "published")
        r2 = await h.query_all()
        check("混合压实后仍为 6 行", r2["matched_rows"] == 6)
        unknown = [x for x in r2["rows"] if x["source_file_id"] is None]
        known = [x for x in r2["rows"] if x["source_file_id"] == legacy_batch_id]
        check(
            "压实后来源未知的仍是原来的 3 行",
            len(unknown) == 3 and all(x["source_row_no"] is None for x in unknown),
        )
        check(
            "旧批次件恢复出的 3 行来源在压实后保持不变",
            len(known) == 3 and sorted(x["source_row_no"] for x in known) == [1, 2, 3],
        )
    finally:
        await h.stop()


async def test_mixed_legacy_and_new_compaction() -> None:
    print("[L5] 新旧文件混合压实：新行来源精确、旧未知行不串批次")
    s = make_settings("l5_mixed")
    h = Harness(s)
    await h.start()
    try:
        # 一个新上传批次（5 列，带来源）
        up = await h.upload([("WTG-01", ts(8, j), float(j)) for j in range(3)])

        # 手工放一个旧三列压实件（来源未知）
        legacy_id = ps.new_file_id("compact")
        legacy_table = pa.table(
            {
                "turbine": pa.array(["WTG-09", "WTG-09"], type=pa.string()),
                "ts_utc": pa.array(
                    [
                        datetime(2026, 11, 13, 7, 0, 0),
                        datetime(2026, 11, 13, 7, 1, 0),
                    ],
                    type=pa.timestamp("ns", tz="UTC"),
                ),
                "vibration": pa.array([9.1, 9.2], type=pa.float64()),
            },
            schema=ps.LEGACY_SCHEMA,
        )
        rel = ps.final_path(DAY, legacy_id)
        os.makedirs(os.path.join(s.lake_dir, f"day={DAY}"), exist_ok=True)
        pq.write_table(legacy_table, os.path.join(s.lake_dir, rel))
        await h.db.insert_file(legacy_id, DAY, rel, 2, ["WTG-09"])

        job = await h.compact()
        check("新旧混合压实成功", job["status"] == "published")
        r = await h.query_all()
        check("总行数 5 = 新 3 + 旧 2", r["matched_rows"] == 5)

        new_rows = [x for x in r["rows"] if x["turbine"] == "WTG-01"]
        old_rows = [x for x in r["rows"] if x["turbine"] == "WTG-09"]
        check(
            "新批次 3 行来源精确指向 up，行号 1..3",
            {(x["source_file_id"], x["source_row_no"]) for x in new_rows}
            == {(up["file_id"], i) for i in range(1, 4)},
        )
        check(
            "旧压实件 2 行保持来源未知，没有借用新批次 ID",
            all(x["source_file_id"] is None and x["source_row_no"] is None for x in old_rows)
            and not any(x["source_file_id"] == up["file_id"] for x in old_rows),
            str(old_rows),
        )
    finally:
        await h.stop()


async def test_snapshot_keeps_lineage_during_reaping() -> None:
    print("[L6] 查询快照横跨压实与回收：来源解析不被旧件回收打断")
    s = make_settings("l6_snapshot")
    h = Harness(s)
    await h.start()
    try:
        up1 = await h.upload([("WTG-01", ts(8, j), float(j)) for j in range(3)])
        up2 = await h.upload([("WTG-02", ts(9, j), float(j + 5)) for j in range(3)])

        # 长持有快照：固定两个批次件
        snap_id, files = await h.db.create_snapshot(None, DAY, DAY)
        check("快照固定 2 个批次件", len(files) == 2)
        snap_refs = [(f["id"], f["path"]) for f in files]

        await h.compact()  # 旧件转 superseded，但被快照保护不会回收
        # 通过快照引用直接读取（模拟慢查询仍在读旧件）
        rows, total, _ = await h.cm.run_io(
            ps.read_filtered, s.lake_dir, snap_refs, None, None, None, 50000
        )
        check("快照内旧件读取仍为 6 行", total == 6)
        check(
            "快照内每行来源仍精确到原批次（up1/up2 各 3 行）",
            {x["source_file_id"] for x in rows} == {up1["file_id"], up2["file_id"]}
            and all(x["source_row_no"] in (1, 2, 3) for x in rows),
        )

        # 释放快照，等待旧件被回收
        await h.db.release_snapshot(snap_id)
        for _ in range(30):
            stats = await h.svc.stats()
            if stats["totals"]["superseded_files"] == 0:
                break
            await asyncio.sleep(0.1)
        stats = await h.svc.stats()
        check("旧件已回收（deleted=2）", stats["totals"]["deleted_files"] == 2, str(stats["totals"]))

        # 回收后新查询走压实件，来源依旧完整
        r = await h.query_all()
        check("回收后 6 行数据可查且来源不丢", r["matched_rows"] == 6)
        check(
            "回收后来源仍指向两个原始上传批次",
            {x["source_file_id"] for x in r["rows"]} == {up1["file_id"], up2["file_id"]},
        )
        for rel in (snap_refs[0][1], snap_refs[1][1]):
            check(f"旧件已不在磁盘: {rel[:24]}…", not os.path.exists(os.path.join(s.lake_dir, rel)))
    finally:
        await h.stop()


async def test_duplicate_rows_distinct_sources_across_uploads() -> None:
    print("[L7] 跨上传的相同内容记录：来源不同，压实不合并来源")
    s = make_settings("l7_dupes")
    h = Harness(s)
    await h.start()
    try:
        payload = [("WTG-01", ts(8, 0), 5.55)]
        up1 = await h.upload(payload, name="a.csv")
        up2 = await h.upload(payload, name="b.csv")
        up3 = await h.upload(payload, name="c.csv")
        await h.compact()
        r = await h.query_all()
        check("三条相同内容记录全部保留", r["matched_rows"] == 3)
        sources = sorted(x["source_file_id"] for x in r["rows"])
        check(
            "三条记录分属三次上传（a/b/c 各一条，行号均为 1）",
            sources == sorted([up1["file_id"], up2["file_id"], up3["file_id"]])
            and all(x["source_row_no"] == 1 for x in r["rows"]),
            str(sources),
        )
        # 校验 JSON 可序列化（API 返回要求）
        import json
        dumped = json.dumps(r["rows"], ensure_ascii=False)
        check("查询结果可 JSON 序列化（含 null 来源）", "source_file_id" in dumped)
    finally:
        await h.stop()


async def main() -> int:
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)
    await test_lineage_on_upload()
    await test_blank_lines_not_counted()
    await test_lineage_survives_multi_round_compaction()
    await test_legacy_three_column_files()
    await test_mixed_legacy_and_new_compaction()
    await test_snapshot_keeps_lineage_during_reaping()
    await test_duplicate_rows_distinct_sources_across_uploads()
    print("\n" + "=" * 56)
    print(f"血缘测试通过 {PASSED} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for f in FAILED:
            print("  FAIL:", f)
        return 1
    print("全部血缘断言通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
