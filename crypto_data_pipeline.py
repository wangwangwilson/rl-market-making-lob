import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tardis_dev import datasets


FEATURE_COLUMNS = [
    "Bid Price 1",
    "Bid Size 1",
    "Ask Price 1",
    "Ask Size 1",
    "midprice",
    "spread",
    "log_return",
    "RV_5min",
    "RSI_5min",
    "OSI_10s",
]


def _normalize_series(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    std = std if std and not np.isclose(std, 0.0) else 1.0
    return (series - mean) / std


def _minmax_series(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    denom = max_val - min_val
    denom = denom if denom and not np.isclose(denom, 0.0) else 1.0
    return (series - min_val) / denom


def _compute_time_based_rsi(midprice: pd.Series, window: str = "5min") -> pd.Series:
    delta = midprice.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))


def _enforce_datetime(ts: pd.Series) -> pd.Series:
    if np.issubdtype(ts.dtype, np.datetime64):
        idx = pd.to_datetime(ts, utc=True)
    else:
        idx = pd.to_datetime(ts, unit="us", utc=True)
    return idx


def _load_raw_frames(
    exchange: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    download_dir: Path,
    data_type: str,
) -> pd.DataFrame:
    symbol_key = symbol.replace(":", "-").replace("/", "-").upper()
    current = start.normalize()
    frames = []

    while current <= end.normalize():
        file_path = download_dir / f"{exchange}_{data_type}_{current.strftime('%Y-%m-%d')}_{symbol_key}.csv.gz"
        if not file_path.exists():
            raise FileNotFoundError(f"未找到原始数据文件: {file_path}")
        frame = pd.read_csv(file_path)
        frames.append(frame)
        current += timedelta(days=1)

    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = _enforce_datetime(raw["timestamp"])
    raw = raw[(raw["timestamp"] >= start) & (raw["timestamp"] <= end)]
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    return raw


def _prepare_features(
    raw: pd.DataFrame,
    resample: str,
    max_rows: Optional[int],
) -> pd.DataFrame:
    df = raw.copy()
    df = df.rename(
        columns={
            "bid_price": "Bid Price 1",
            "bid_amount": "Bid Size 1",
            "ask_price": "Ask Price 1",
            "ask_amount": "Ask Size 1",
        }
    )

    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")]

    if resample:
        df = df.resample(resample).last().ffill()

    df["midprice"] = (df["Bid Price 1"] + df["Ask Price 1"]) / 2
    df["spread"] = df["Ask Price 1"] - df["Bid Price 1"]
    df["log_return"] = np.log(df["midprice"]).diff()
    df["RV_5min"] = df["log_return"].rolling("5min", min_periods=1).std().fillna(0.0)
    df["RSI_5min"] = _compute_time_based_rsi(df["midprice"], "5min")

    volume_sum = df["Bid Size 1"] + df["Ask Size 1"]
    volume_sum = volume_sum.replace(0, np.nan)
    df["OSI_10s"] = ((df["Bid Size 1"] - df["Ask Size 1"]) / volume_sum).rolling("10s", min_periods=1).mean()
    df["OSI_10s"] = df["OSI_10s"].fillna(0.0)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if max_rows is not None:
        df = df.iloc[:max_rows]

    # Scaling
    zscore_cols = ["Bid Price 1", "Ask Price 1", "midprice", "spread", "log_return", "RV_5min", "RSI_5min"]
    minmax_cols = ["Bid Size 1", "Ask Size 1", "OSI_10s"]

    for col in zscore_cols:
        df[col] = _normalize_series(df[col])
    for col in minmax_cols:
        df[col] = _minmax_series(df[col])

    df = df[FEATURE_COLUMNS]
    return df.astype(np.float32)


def prepare_crypto_dataset(
    exchange: str,
    symbol: str,
    start: str,
    end: str,
    api_key: Optional[str] = None,
    download_dir: str = "data/tardis_raw",
    data_type: str = "book_ticker",
    resample: str = "1s",
    max_rows: Optional[int] = 5000,
) -> pd.DataFrame:
    """
    下载并处理 Tardis 高频数据，生成策略所需特征。
    """
    api_key = api_key or os.getenv("TARDIS_API_KEY")
    if not api_key:
        raise ValueError("需要提供 Tardis API Key，可通过参数或环境变量 TARDIS_API_KEY 设置。")

    download_dir_path = Path(download_dir)
    download_dir_path.mkdir(parents=True, exist_ok=True)

    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if end_ts <= start_ts:
        raise ValueError("结束时间必须晚于开始时间")

    download_start = start_ts.normalize()
    download_end = (end_ts.normalize() + pd.Timedelta(days=1))

    datasets.download(
        exchange=exchange,
        data_types=[data_type],
        symbols=[symbol],
        from_date=download_start.strftime("%Y-%m-%d"),
        to_date=download_end.strftime("%Y-%m-%d"),
        format="csv",
        api_key=api_key,
        download_dir=str(download_dir_path),
    )

    raw = _load_raw_frames(exchange, symbol, start_ts, end_ts, download_dir_path, data_type)
    return _prepare_features(raw, resample=resample, max_rows=max_rows)

