"""Parquet 文件层。

关键约束：本模块所有函数只做磁盘 / PyArrow 操作，不打开数据库事务，
因此“读写 Parquet 不持数据库写锁”。调用方先在 DB 短事务之外完成文件 IO，
再用 DB 短事务登记元数据。

文件布局：
  lake/day=YYYY-MM-DD/<id>.parquet        已发布的日分区文件（批次件 / 压实件）
  lake/_tmp/<id>.parquet                  尚未发布的临时件（崩溃后重启清理）

数据血缘（lineage）：
- 新文件为 5 列 schema，每个读数携带 source_file_id（原 CSV 上传批次 ID）
  与 source_row_no（该 CSV 中第几条非空数据行，不含表头，从 1 开始）；
- 压实只做按行拼接，血缘列原样保留，多轮压实后仍指向最初的上传批次，
  永远不会变成压实合并件的 ID；
- 历史遗留的三列 Parquet 仍可读，读取时按文件 ID 前缀补血缘：
  batch- 批次件按“本文件 + 行内次序”恢复，compact- 压实件无法判定，
  两列均补 null（上层展示为“来源未知”），且未知行不会被记到其他批次名下。
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# 基础三列：机组号、UTC 时间、振动值（均为非空有限值）
BASE_FIELDS = [
    pa.field("turbine", pa.string(), nullable=False),
    pa.field("ts_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
    pa.field("vibration", pa.float64(), nullable=False),
]
# 血缘两列：可空 —— 旧压实件无法判定来源时为 null（“来源未知”）
LINEAGE_FIELDS = [
    pa.field("source_file_id", pa.string(), nullable=True),
    pa.field("source_row_no", pa.int64(), nullable=True),
]

# 新写入文件的完整 5 列 schema
SCHEMA = pa.schema(BASE_FIELDS + LINEAGE_FIELDS)

# 历史遗留文件的三列 schema（仅用于识别 / 测试构造旧件）
LEGACY_SCHEMA = pa.schema(BASE_FIELDS)

# 文件引用：(文件 ID, 相对 lake_dir 的路径)。文件 ID 用于旧三列件的血缘恢复。
FileRef = tuple[str, str]


@dataclass(frozen=True)
class Reading:
    turbine: str
    ts_utc: datetime
    vibration: float
    # 该读数在原 CSV 中是第几条非空数据行（不含表头，从 1 开始）
    source_row_no: int


def new_file_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def file_id_from_path(rel_path: str) -> str:
    """从相对路径取文件 ID（文件名去掉 .parquet 后缀）。"""
    return Path(rel_path).stem


def _day_dir(lake_dir: str, day: str) -> str:
    return os.path.join(lake_dir, f"day={day}")


def final_path(day: str, file_id: str) -> str:
    return os.path.join(f"day={day}", f"{file_id}.parquet")


def temp_path(file_id: str) -> str:
    return os.path.join("_tmp", f"{file_id}.parquet")


def _abs(lake_dir: str, rel_path: str) -> str:
    return os.path.join(lake_dir, rel_path)


def ensure_dirs(lake_dir: str) -> None:
    os.makedirs(os.path.join(lake_dir, "_tmp"), exist_ok=True)


def _attach_lineage(table: pa.Table, file_id: str) -> pa.Table:
    """把任意表规整为 5 列 SCHEMA，并为旧三列件补血缘。

    - 表中已含血缘两列（新批次件 / 新压实件）：原样保留，含 null 的“未知”行
      也逐行保留，绝不因为压实而被改写或挂到其他批次名下；
    - 旧三列批次件（ID 以 batch- 开头）：一批就是一次上传，按行内次序恢复
      source_file_id=本文件 ID、source_row_no=1..N；
    - 其他旧三列件（典型为 compact- 压实件，以及任何无法确认身份的文件）：
      保守处理为两列 null（来源未知），避免错误归属。
    """
    n = table.num_rows
    has_lineage = "source_file_id" in table.column_names and "source_row_no" in table.column_names
    if has_lineage:
        return table.select([f.name for f in SCHEMA]).cast(SCHEMA)

    if file_id.startswith("batch-"):
        source_ids = pa.array([file_id] * n, type=pa.string())
        row_nos = pa.array(range(1, n + 1), type=pa.int64())
    else:
        # 旧压实件混合了多个批次且无内嵌血缘，无法判定 → 来源未知
        source_ids = pa.nulls(n, type=pa.string())
        row_nos = pa.nulls(n, type=pa.int64())

    base = table.select([f.name for f in BASE_FIELDS])
    return pa.table(
        {
            "turbine": base.column("turbine"),
            "ts_utc": base.column("ts_utc"),
            "vibration": base.column("vibration"),
            "source_file_id": source_ids,
            "source_row_no": row_nos,
        },
        schema=SCHEMA,
    )


def write_batch(lake_dir: str, day: str, file_id: str, readings: Sequence[Reading]) -> str:
    """把一整批读数写成一个 Parquet 文件，返回相对路径（先落 _tmp，再原子改名）。

    新上传批次的每行血缘：source_file_id 即本次上传的批次文件 ID，
    source_row_no 为 CSV 非空数据行序号；重复内容的读数各自保留独立来源。
    """
    ensure_dirs(lake_dir)
    table = pa.table(
        {
            "turbine": pa.array([r.turbine for r in readings], type=pa.string()),
            "ts_utc": pa.array(
                [r.ts_utc.astimezone(timezone.utc) for r in readings],
                type=pa.timestamp("ns", tz="UTC"),
            ),
            "vibration": pa.array([r.vibration for r in readings], type=pa.float64()),
            "source_file_id": pa.array([file_id] * len(readings), type=pa.string()),
            "source_row_no": pa.array(
                [r.source_row_no for r in readings], type=pa.int64()
            ),
        },
        schema=SCHEMA,
    )
    rel_tmp = temp_path(file_id)
    rel_final = final_path(day, file_id)
    os.makedirs(_day_dir(lake_dir, day), exist_ok=True)
    pq.write_table(table, _abs(lake_dir, rel_tmp), compression="zstd")
    os.replace(_abs(lake_dir, rel_tmp), _abs(lake_dir, rel_final))
    return rel_final


def write_temp(lake_dir: str, file_id: str, table: pa.Table) -> str:
    """压实合并结果先写 _tmp（发布前崩溃不影响已提交数据）。"""
    ensure_dirs(lake_dir)
    rel_tmp = temp_path(file_id)
    pq.write_table(table, _abs(lake_dir, rel_tmp), compression="zstd")
    return rel_tmp


def publish_temp(lake_dir: str, day: str, file_id: str, rel_tmp: str) -> str:
    """临时件原子改名进入日分区。"""
    rel_final = final_path(day, file_id)
    os.makedirs(_day_dir(lake_dir, day), exist_ok=True)
    os.replace(_abs(lake_dir, rel_tmp), _abs(lake_dir, rel_final))
    return rel_final


def _read_one(lake_dir: str, file_id: str, rel_path: str) -> pa.Table:
    """读取单个 Parquet 并规整为 5 列（兼容旧三列件）。"""
    table = pq.read_table(_abs(lake_dir, rel_path))
    return _attach_lineage(table, file_id)


def read_filtered(
    lake_dir: str,
    file_refs: Iterable[FileRef],
    turbine: str | None,
    start: datetime | None,
    end: datetime | None,
    limit: int,
) -> tuple[list[dict], int, int]:
    """读取给定文件集合（由读快照固定），按机组 / 时段下推过滤。

    重复行原样保留（pyarrow 不做去重），每行同时返回来源文件 ID 与行序号；
    旧压实件无法判定的行两字段为 null。返回 (行, 命中总数, 实际扫描文件数)。
    """
    frames: list[pa.Table] = []
    scanned = 0
    for file_id, rel in file_refs:
        table = _read_one(lake_dir, file_id, rel)
        scanned += 1
        mask = None
        if turbine is not None:
            mask = pc.equal(table["turbine"], turbine)
        if start is not None:
            m = pc.greater_equal(table["ts_utc"], pa.scalar(start, type=table["ts_utc"].type))
            mask = m if mask is None else pc.and_(mask, m)
        if end is not None:
            m = pc.less(table["ts_utc"], pa.scalar(end, type=table["ts_utc"].type))
            mask = m if mask is None else pc.and_(mask, m)
        if mask is not None:
            table = table.filter(mask)
        if table.num_rows:
            frames.append(table)

    if not frames:
        return [], 0, scanned
    combined = pa.concat_tables(frames, promote_options="default")
    total = combined.num_rows
    if total > limit:
        combined = combined.slice(0, limit)
    rows = [
        {
            "turbine": t,
            "ts_utc": ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "vibration": v,
            "source_file_id": src_id,
            "source_row_no": src_row,
        }
        for t, ts, v, src_id, src_row in zip(
            combined.column("turbine").to_pylist(),
            combined.column("ts_utc").to_pylist(),
            combined.column("vibration").to_pylist(),
            combined.column("source_file_id").to_pylist(),
            combined.column("source_row_no").to_pylist(),
        )
    ]
    return rows, total, scanned


def merge_files(lake_dir: str, file_refs: Sequence[FileRef]) -> pa.Table:
    """按给定顺序拼接多个 Parquet，保留全部重复行与逐行血缘。

    新旧文件混合压实时，每个文件先独立规整血缘：旧批次件恢复为本文件来源，
    旧压实件补 null（来源未知），因此未知行不会被误记到任何其他批次。
    """
    tables = [_read_one(lake_dir, file_id, rel) for file_id, rel in file_refs]
    if not tables:
        return pa.table({f.name: [] for f in SCHEMA}, schema=SCHEMA)
    return pa.concat_tables(tables, promote_options="default")


def delete_file(lake_dir: str, rel_path: str) -> None:
    try:
        os.remove(_abs(lake_dir, rel_path))
    except FileNotFoundError:
        # 恢复场景下可能已被回收，幂等处理
        pass


def list_disk_paths(lake_dir: str) -> set[str]:
    found: set[str] = set()
    for root, _dirs, files in os.walk(lake_dir):
        for name in files:
            if not name.endswith(".parquet"):
                continue
            full = os.path.join(root, name)
            found.add(os.path.relpath(full, lake_dir))
    return found


def list_temp_paths(lake_dir: str) -> list[str]:
    tmp_dir = os.path.join(lake_dir, "_tmp")
    if not os.path.isdir(tmp_dir):
        return []
    return [
        os.path.join("_tmp", name)
        for name in os.listdir(tmp_dir)
        if name.endswith(".parquet")
    ]


def table_turbines(table: pa.Table) -> list[str]:
    return sorted(set(table.column("turbine").to_pylist()))
