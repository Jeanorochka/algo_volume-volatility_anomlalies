import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import apimoex
import pandas as pd

# === НАСТРОЙКИ ===

DATA_DIR = "history_5m"     # папка с историей по всем акциям
MAX_WORKERS = 8             # сколько тикеров качать параллельно
MONTHS_BACK = 6             # сколько месяцев истории держим
BOARD = "TQBR"              # основной режим акций

# 1-минутные свечи с MOEX
CANDLE_COLUMNS = ("begin", "open", "high", "low", "close", "volume")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def dt_now_msk_date():
    # просто берем локальную дату машины
    return datetime.now().date()


def global_start_date():
    """Дата с которой нужен максимум 6 месяцев истории."""
    today = dt_now_msk_date()
    # грубо: 30 дней * MONTHS_BACK
    approx_days = 30 * MONTHS_BACK
    start = today - timedelta(days=approx_days)
    return start


def get_all_tqbr_tickers():
    """
    Тянем список всех бумаг с TQBR.
    """
    url = (
        f"https://iss.moex.com/iss/engines/stock/"
        f"markets/shares/boards/{BOARD}/securities.json"
    )
    arguments = {"securities.columns": "SECID,SHORTNAME"}

    with requests.Session() as session:
        iss = apimoex.ISSClient(session, url, arguments)
        data = iss.get()

    df = pd.DataFrame(data["securities"], columns=["SECID", "SHORTNAME"])
    tickers = df["SECID"].dropna().unique().tolist()
    return tickers


def load_1m_from_moex(ticker: str, start_date: str) -> pd.DataFrame:
    """
    Качаем 1-минутки по тикеру с заданной даты.
    """
    with requests.Session() as session:
        candles = apimoex.get_board_candles(
            session,
            security=ticker,
            interval=1,              # 1 минута
            start=start_date,
            end=None,
            columns=CANDLE_COLUMNS,
        )

    df = pd.DataFrame(candles)
    if df.empty:
        return df

    df["begin"] = pd.to_datetime(df["begin"])
    df.set_index("begin", inplace=True)
    # сортируем на всякий случай
    df.sort_index(inplace=True)
    return df


def resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    1m -> 5m OHLCV.
    """
    if df_1m.empty:
        return df_1m

    df_5m = df_1m.resample("5T", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    # выбрасываем бары без сделок
    df_5m.dropna(subset=["open", "high", "low", "close"], inplace=True)
    return df_5m


def history_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker}_5m.csv")


def load_existing_history(ticker: str) -> pd.DataFrame:
    path = history_path(ticker)
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


def save_history(ticker: str, df_5m: pd.DataFrame):
    path = history_path(ticker)
    df_5m.to_csv(path, encoding="utf-8", index_label="datetime")


def update_ticker_history(ticker: str) -> str:
    """
    Обновляем историю 5m по одному тикеру:
    - читаем старую историю, если есть
    - считаем, с какой даты докачивать 1m
    - качаем, ресемплим, мерджим, режем 6 месяцев
    """
    try:
        existing = load_existing_history(ticker)

        if existing.empty:
            # нет истории — качаем с глобальной стартовой
            start_date = global_start_date()
        else:
            # берем последнюю дату и откатываемся на день назад,
            # чтобы перекрыть возможные дырки
            last_dt = existing.index.max()
            start_date = (last_dt.date() - timedelta(days=1))

        start_date_str = start_date.isoformat()

        df_1m = load_1m_from_moex(ticker, start_date_str)
        if df_1m.empty:
            return f"{ticker}: нет свежих данных с {start_date_str}"

        df_new_5m = resample_to_5m(df_1m)

        if existing.empty:
            combined = df_new_5m
        else:
            combined = pd.concat([existing, df_new_5m])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)

        # режем до 6 месяцев
        cutoff = global_start_date()
        combined = combined[combined.index.date >= cutoff]

        save_history(ticker, combined)

        return f"{ticker}: баров всего={len(combined)}, новых={len(df_new_5m)}"

    except Exception as e:
        return f"{ticker}: ОШИБКА {e}"


def main():
    ensure_data_dir()

    tickers = get_all_tqbr_tickers()
    print(f"Тикеров на {BOARD}: {len(tickers)}")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(update_ticker_history, t): t for t in tickers
        }

        for i, future in enumerate(as_completed(future_to_ticker), start=1):
            msg = future.result()
            results.append(msg)
            print(f"[{i}/{len(tickers)}] {msg}")

    # при желании можно сохранить лог
    log_path = os.path.join(DATA_DIR, "update_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print("\nГотово. Логи в", log_path)


if __name__ == "__main__":
    main()
