import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import apimoex
import pandas as pd

# === НАСТРОЙКИ ===

# Папка с ИСТОРИЕЙ 1-МИНУТНЫХ свечей по всем акциям
DATA_DIR = "history_1m"     # важно: совпадает с HISTORY_DIR у робота
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
    """Дата, с которой нужен максимум MONTHS_BACK месяцев истории."""
    today = dt_now_msk_date()
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
    df.sort_index(inplace=True)
    # дальше индекс = время свечи, его мы будем сохранять как "datetime"
    return df


def history_path(ticker: str) -> str:
    # формат имени файла: TICKER_1m.csv
    return os.path.join(DATA_DIR, f"{ticker}_1m.csv")


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


def save_history(ticker: str, df_1m: pd.DataFrame):
    """
    Сохраняем 1-минутки в CSV в формате:
    колонка datetime + open/high/low/close/volume.
    """
    path = history_path(ticker)
    # индекс сохраняем как колонку "datetime" — так его потом читает робот
    df_1m.to_csv(path, encoding="utf-8", index_label="datetime")


def update_ticker_history(ticker: str) -> str:
    """
    Обновляем историю 1m по одному тикеру:
    - читаем старую историю, если есть
    - считаем, с какой даты докачивать 1m
    - качаем, мерджим, режем до MONTHS_BACK месяцев
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

        df_1m_new = load_1m_from_moex(ticker, start_date_str)
        if df_1m_new.empty:
            return f"{ticker}: нет свежих данных с {start_date_str}"

        # комбинируем старую и новую историю 1m
        if existing.empty:
            combined = df_1m_new
        else:
            combined = pd.concat([existing, df_1m_new])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)

        # режем до нужного горизонта (MONTHS_BACK)
        cutoff = global_start_date()
        combined = combined[combined.index.date >= cutoff]

        save_history(ticker, combined)

        return f"{ticker}: баров всего={len(combined)}, новых={len(df_1m_new)}"

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
