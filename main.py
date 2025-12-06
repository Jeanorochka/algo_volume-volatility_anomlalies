import os
import csv
import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Tuple, Optional
from datetime import datetime

import pandas as pd
from tinkoff.invest import (
    Client,
    CandleInstrument,
    SubscriptionInterval,
    OrderDirection,
    OrderType,
)
from tinkoff.invest.services import MarketDataStreamManager
from tinkoff.invest.utils import quotation_to_decimal
from tinkoff.invest.exceptions import InvestError

# ================== КОНФИГ ==================

# Имена переменных окружения
TOKEN_ENV_NAME = "INVEST_TOKEN"
ACCOUNT_ID_ENV_NAME = "INVEST_ACCOUNT_ID"

# Папка с минутной историей по всем акциям (из update_history_all_1m.py, например)
HISTORY_DIR = "history_1m"
HISTORY_FILE_SUFFIX = "_1m.csv"

# Файл конфигурации с пер-тикерными настройками
CONFIG_FILE = "config.json"

# Число инструментов
MAX_INSTRUMENTS = 170

# БАЗОВЫЕ параметры аномалий (z-score), от них пляшем по умолчанию
# Чуть снизил vol-порог, чтобы входить немного раньше по объёму
BASE_VOL_Z_ENTRY = 2.2   # было 2.5
BASE_RET_Z_ENTRY = 1.4   # было 1.5

# Базовые параметры выхода (могут быть переопределены на уровне тикера)
TAKE_PROFIT_PCT = 0.0025   # +0.25% профит
STOP_LOSS_PCT   = -0.0015  # -0.15% убыток

# Базовый объём сделки
LOTS_PER_TRADE = 1

# Безопасность: True = НЕ шлём ордера, только логируем
DRY_RUN = True

# Минимальное число баров в истории, чтобы тикер считался валидным
MIN_BARS_HISTORY = 200

# Комиссия брокера на одну сторону сделки, доля от объёма
COMMISSION_PER_SIDE_PCT = 0.00004

# Дневной лимит убытка (по сумме net-PnL с начала запуска), доля (−0.5 = −50%)
DAILY_LOSS_LIMIT_PCT = -0.5

# Лимиты по числу позиций
MAX_OPEN_POSITIONS_TOTAL = 30          # максимум одновременно открытых позиций
MAX_OPEN_POSITIONS_PER_SECTOR = 3      # максимум позиций в одном секторе

# Файл лога сделок (общий для всех тикеров)
TRADES_LOG_FILE = "trades_log.csv"

# Тикеры, которые НЕ хотим торговать вообще
EXCLUDED_TICKERS = {"BANE", "YDEX", "BANEP"}


# ================== СОСТОЯНИЕ ==================

@dataclass
class InstrumentState:
    vol_mean: float
    vol_std: float
    ret_mean: float
    ret_std: float
    last_price: Decimal

    # Пороговые значения z-score для этого тикера (адаптивные + конфиг)
    vol_z_entry: float = BASE_VOL_Z_ENTRY
    ret_z_entry: float = BASE_RET_Z_ENTRY

    in_position: bool = False
    entry_price: Decimal = Decimal("0")

    # Пер-тикерные торговые параметры
    lots_per_trade: int = LOTS_PER_TRADE
    take_profit_pct: float = TAKE_PROFIT_PCT
    stop_loss_pct: float = STOP_LOSS_PCT
    enabled: bool = True  # можно отключить тикер через конфиг

    # НОВОЕ: максимальный допустимый спред в долях (0.005 = 0.5%)
    max_spread_pct: float = 0.005


@dataclass
class RiskState:
    # сумма net-PnL по закрытым сделкам с момента запуска (в долях, 0.01 = +1%)
    daily_realized_pnl: float = 0.0
    # общее число открытых позиций
    open_positions_total: int = 0
    # количество позиций по секторам: sector -> count
    open_by_sector: Dict[str, int] = field(default_factory=dict)


# ================== ВСПОМОГАТЕЛЬНОЕ ==================

def calc_mean_std(series: pd.Series) -> Tuple[float, float]:
    """Возвращает (mean, std), защищённый от std=0."""
    if series is None or len(series) == 0:
        return 0.0, 1.0
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if std == 0.0:
        std = 1.0
    return mean, std


def calc_z_score(x: float, mean: float, std: float) -> float:
    if std == 0.0:
        return 0.0
    return (x - mean) / std


def moneyvalue_to_decimal(mv) -> Decimal:
    """
    Локальный аналог moneyvalue_to_decimal для старых версий tinkoff.invest.
    MoneyValue -> Decimal.
    """
    if mv is None:
        return Decimal("0")
    return Decimal(mv.units) + (Decimal(mv.nano) / Decimal(1_000_000_000))


def get_current_spread_pct(client: Client, figi: str) -> Optional[float]:
    """
    Получаем текущий спред из стакана (depth=1) и возвращаем в долях:
    (ask - bid) / mid. Если нет данных — возвращаем None.
    """
    try:
        ob = client.market_data.get_order_book(figi=figi, depth=1)
    except Exception as e:
        logging.error("Не удалось получить стакан для figi=%s: %s", figi, e)
        return None

    if not ob.bids or not ob.asks:
        return None

    best_bid = moneyvalue_to_decimal(ob.bids[0].price)
    best_ask = moneyvalue_to_decimal(ob.asks[0].price)

    if best_bid <= 0 or best_ask <= 0:
        return None

    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    if mid == 0:
        return None

    spread_pct = float(spread / mid)
    return spread_pct


# ================== ЗАГРУЗКА ИСТОРИИ ==================

def load_history_for_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """Читаем HISTORY_DIR/<TICKER>_1m.csv, если есть."""
    path = os.path.join(HISTORY_DIR, f"{ticker}{HISTORY_FILE_SUFFIX}")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or "datetime" not in df.columns:
        return None

    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df


def build_states_from_history() -> Dict[str, InstrumentState]:
    """
    Проходим по всем файлам HISTORY_DIR/*_1m.csv, считаем статистику
    и возвращаем словарь: тикер -> InstrumentState.
    """
    states: Dict[str, InstrumentState] = {}

    if not os.path.isdir(HISTORY_DIR):
        logging.error("Папка с историей %s не найдена", HISTORY_DIR)
        return states

    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(HISTORY_FILE_SUFFIX):
            continue

        ticker = fname.split("_")[0]

        df = load_history_for_ticker(ticker)
        if df is None or len(df) < MIN_BARS_HISTORY:
            continue

        vol_series = df["volume"].astype(float)
        close_series = df["close"].astype(float)
        ret_series = close_series.pct_change().dropna()

        if len(ret_series) < MIN_BARS_HISTORY:
            continue

        vol_mean, vol_std = calc_mean_std(vol_series)
        ret_mean, ret_std = calc_mean_std(ret_series)
        last_price = Decimal(str(close_series.iloc[-1]))

        states[ticker] = InstrumentState(
            vol_mean=vol_mean,
            vol_std=vol_std,
            ret_mean=ret_mean,
            ret_std=ret_std,
            last_price=last_price,
        )

    if not states:
        logging.error("Не удалось построить состояния ни для одного тикера из истории.")
        return states

    logging.info("По истории построено состояний: %s тикеров", len(states))

    # --- АДАПТИВНЫЕ ПОРОГИ ДЛЯ КАЖДОГО ТИКЕРА ---

    vol_stds = [s.vol_std for s in states.values() if s.vol_std > 0]
    ret_stds = [s.ret_std for s in states.values() if s.ret_std > 0]

    global_vol_std = sum(vol_stds) / len(vol_stds) if vol_stds else 1.0
    global_ret_std = sum(ret_stds) / len(ret_stds) if ret_stds else 1.0

    for ticker, st in states.items():
        # коэффициенты волатильности относительно среднего по всем тикерам
        vol_scale = st.vol_std / global_vol_std if global_vol_std > 0 else 1.0
        ret_scale = st.ret_std / global_ret_std if global_ret_std > 0 else 1.0

        # ограничим, чтобы не было экстремального безумия
        vol_scale = max(0.5, min(2.0, vol_scale))
        ret_scale = max(0.5, min(2.0, ret_scale))

        st.vol_z_entry = BASE_VOL_Z_ENTRY * vol_scale
        st.ret_z_entry = BASE_RET_Z_ENTRY * ret_scale

    return states


# ================== РАБОТА С CONFIG.JSON ==================

def load_or_init_config(path: str) -> dict:
    """Читаем config.json, если нет или битый — возвращаем пустой каркас."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict):
                logging.warning("Файл %s не является JSON-объектом, переинициализируем.", path)
                return {"tickers": {}}
            if "tickers" not in cfg or not isinstance(cfg["tickers"], dict):
                cfg["tickers"] = {}
            return cfg
        except Exception as e:
            logging.warning("Не удалось прочитать %s: %s. Используем пустой конфиг.", path, e)
            return {"tickers": {}}
    else:
        logging.info("Файл %s не найден, создадим новый конфиг с дефолтами.", path)
        return {"tickers": {}}


def ensure_config_has_all_tickers(
    config: dict,
    states: Dict[str, InstrumentState],
    path: str
) -> dict:
    """
    Гарантируем, что в конфиге есть запись для каждого тикера из states.
    Если чего-то нет — добавляем с дефолтами и сразу сохраняем.
    """
    changed = False
    tick_cfg = config.setdefault("tickers", {})
    for ticker, st in states.items():
        if ticker not in tick_cfg:
            tick_cfg[ticker] = {
                "enabled": True,
                "lots_per_trade": LOTS_PER_TRADE,
                "vol_z_entry": st.vol_z_entry,
                "ret_z_entry": st.ret_z_entry,
                "take_profit_pct": TAKE_PROFIT_PCT,
                "stop_loss_pct": STOP_LOSS_PCT,
                # НОВОЕ: дефолтный лимит по спреду (0.5%)
                "max_spread_pct": 0.005,
            }
            changed = True

    if changed:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            logging.info(
                "Обновили %s: добавили %d тикеров без настроек.",
                path,
                sum(1 for _ in states if _ in config["tickers"]),
            )
        except Exception as e:
            logging.error("Не удалось сохранить конфиг %s: %s", path, e)

    return config


def apply_config_to_states(states: Dict[str, InstrumentState], config: dict):
    """
    Применяем настройки из config["tickers"] к InstrumentState.
    """
    tick_cfg = config.get("tickers", {})
    for ticker, st in states.items():
        cfg = tick_cfg.get(ticker)
        if not cfg:
            continue

        st.enabled = bool(cfg.get("enabled", True))

        if "vol_z_entry" in cfg:
            try:
                st.vol_z_entry = float(cfg["vol_z_entry"])
            except (TypeError, ValueError):
                pass

        if "ret_z_entry" in cfg:
            try:
                st.ret_z_entry = float(cfg["ret_z_entry"])
            except (TypeError, ValueError):
                pass

        if "lots_per_trade" in cfg:
            try:
                st.lots_per_trade = int(cfg["lots_per_trade"])
            except (TypeError, ValueError):
                pass

        if "take_profit_pct" in cfg:
            try:
                st.take_profit_pct = float(cfg["take_profit_pct"])
            except (TypeError, ValueError):
                pass

        if "stop_loss_pct" in cfg:
            try:
                st.stop_loss_pct = float(cfg["stop_loss_pct"])
            except (TypeError, ValueError):
                pass

        if "max_spread_pct" in cfg:
            try:
                st.max_spread_pct = float(cfg["max_spread_pct"])
            except (TypeError, ValueError):
                pass


# ================== ИНСТРУМЕНТЫ ТИНЬКОФФ ==================

def get_moex_shares_meta(client: Client):
    """
    Получаем все акции через Tinkoff API и фильтруем RUB-акции,
    доступные к торговле через API.

    ВАЖНО: БЕЗ inst.exchange == "MOEX", чтобы не отрезать половину рынка.
    При желании можно добавить фильтр по country_of_risk == "RU".
    """
    shares = client.instruments.shares().instruments
    logging.info("Всего акций в API: %d", len(shares))

    ticker_to_figi: Dict[str, str] = {}
    ticker_to_sector: Dict[str, Optional[str]] = {}

    for inst in shares:
        if (
            inst.currency.lower() == "rub"
            and inst.api_trade_available_flag
            and inst.buy_available_flag
            # and inst.country_of_risk == "RU"
        ):
            ticker = inst.ticker
            ticker_to_figi[ticker] = inst.figi
            sector = getattr(inst, "sector", "")
            ticker_to_sector[ticker] = sector or None

    logging.info("RUB-акций, доступных к покупке через API: %d", len(ticker_to_figi))
    return ticker_to_figi, ticker_to_sector


# ================== ЛОГ СДЕЛОК ==================

def log_trade(
    ticker: str,
    side: str,          # "BUY" или "SELL"
    price: Decimal,
    lots: int,
    gross_pnl_pct: Optional[float],      # None для входа
    commission_pct: Optional[float],     # None для входа
    net_pnl_pct: Optional[float],        # None для входа
    candle_time,
    dry_run: bool,
    exit_reason: Optional[str] = None,   # НОВОЕ: причина выхода (TP_TOUCH / SL_TOUCH)
):
    """Записываем сделку в CSV. Один общий файл для всех тикеров."""
    ts_log = datetime.utcnow().isoformat()
    candle_time_str = str(candle_time)

    file_exists = os.path.exists(TRADES_LOG_FILE)

    with open(TRADES_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "ts_log_utc",
                "candle_time",
                "ticker",
                "side",
                "price",
                "lots",
                "gross_pnl_pct",
                "commission_pct",
                "net_pnl_pct",
                "dry_run",
                "exit_reason",
            ])
        writer.writerow([
            ts_log,
            candle_time_str,
            ticker,
            side,
            float(price),
            lots,
            None if gross_pnl_pct is None else float(gross_pnl_pct),
            None if commission_pct is None else float(commission_pct),
            None if net_pnl_pct is None else float(net_pnl_pct),
            int(dry_run),
            exit_reason or "",
        ])


# ================== RISK: МОЖНО ЛИ ОТКРЫВАТЬ НОВУЮ ПОЗУ ==================

def can_open_new_position(
    ticker: str,
    sector: Optional[str],
    risk_state: RiskState,
) -> bool:
    # Дневной лимит по net-PnL (в долях)
    if risk_state.daily_realized_pnl <= DAILY_LOSS_LIMIT_PCT:
        logging.warning(
            "Дневной лимит убытка достигнут (%.2f%%). Новые позиции не открываем (тикер %s).",
            risk_state.daily_realized_pnl * 100,
            ticker,
        )
        return False

    # Общий лимит открытых позиций
    if risk_state.open_positions_total >= MAX_OPEN_POSITIONS_TOTAL:
        logging.info(
            "Лимит открытых позиций (%d) достигнут, пропускаем сигнал по %s",
            MAX_OPEN_POSITIONS_TOTAL,
            ticker,
        )
        return False

    # Лимит позиций по сектору
    if sector:
        sector_count = risk_state.open_by_sector.get(sector, 0)
        if sector_count >= MAX_OPEN_POSITIONS_PER_SECTOR:
            logging.info(
                "Лимит позиций в секторе '%s' (%d) достигнут, пропускаем сигнал по %s",
                sector,
                MAX_OPEN_POSITIONS_PER_SECTOR,
                ticker,
            )
            return False

    return True


# ================== ОРДЕРА ==================

def open_long(
    client: Client,
    account_id: str,
    figi: str,
    ticker: str,
    state: InstrumentState,
    candle_time,
    risk_state: RiskState,
    sector: Optional[str],
):
    if state.in_position:
        return

    if risk_state.daily_realized_pnl <= DAILY_LOSS_LIMIT_PCT:
        logging.warning(
            "Дневной лимит убытка уже достигнут (%.2f%%), не открываем %s",
            risk_state.daily_realized_pnl * 100,
            ticker,
        )
        return

    last_price = state.last_price
    lots = state.lots_per_trade

    if DRY_RUN:
        logging.info("[DRY_RUN] BUY %s x%d", ticker, lots)
        state.in_position = True
        state.entry_price = last_price
        log_trade(
            ticker=ticker,
            side="BUY",
            price=last_price,
            lots=lots,
            gross_pnl_pct=None,
            commission_pct=None,
            net_pnl_pct=None,
            candle_time=candle_time,
            dry_run=True,
        )
        risk_state.open_positions_total += 1
        if sector:
            risk_state.open_by_sector[sector] = risk_state.open_by_sector.get(sector, 0) + 1
        return

    from uuid import uuid4

    try:
        resp = client.orders.post_order(
            figi=figi,
            quantity=lots,
            direction=OrderDirection.ORDER_DIRECTION_BUY,
            account_id=account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
            order_id=str(uuid4()),
        )
        logging.info("BUY отправлен: %s x%d, resp=%s", ticker, lots, resp)
        state.in_position = True
        state.entry_price = last_price  # при желании можно взять resp.executed_order_price
        log_trade(
            ticker=ticker,
            side="BUY",
            price=last_price,
            lots=lots,
            gross_pnl_pct=None,
            commission_pct=None,
            net_pnl_pct=None,
            candle_time=candle_time,
            dry_run=False,
        )
        risk_state.open_positions_total += 1
        if sector:
            risk_state.open_by_sector[sector] = risk_state.open_by_sector.get(sector, 0) + 1
    except InvestError as e:
        logging.error("Ошибка BUY по %s: %s", ticker, e)


def close_long(
    client: Client,
    account_id: str,
    figi: str,
    ticker: str,
    state: InstrumentState,
    candle_time,
    risk_state: RiskState,
    sector: Optional[str],
    exit_reason: str,  # НОВОЕ: причина выхода (TP_TOUCH / SL_TOUCH)
):
    if not state.in_position:
        return

    last_price = state.last_price
    entry = state.entry_price if state.entry_price != 0 else last_price
    if entry == 0:
        return

    lots = state.lots_per_trade

    # Ретёрн в долях
    price_ratio = float(last_price / entry)
    ret = price_ratio - 1.0
    gross_pnl_pct = ret * 100.0
    commission_pct = COMMISSION_PER_SIDE_PCT * 2.0 * 100.0
    net_pnl_pct = gross_pnl_pct - commission_pct
    # net-PnL в долях (для риск-менеджмента)
    ret_net = ret - 2.0 * COMMISSION_PER_SIDE_PCT

    if DRY_RUN:
        logging.info(
            "[DRY_RUN] SELL %s x%d, gross=%.2f%%, net=%.2f%%, reason=%s",
            ticker,
            lots,
            gross_pnl_pct,
            net_pnl_pct,
            exit_reason,
        )
        log_trade(
            ticker=ticker,
            side="SELL",
            price=last_price,
            lots=lots,
            gross_pnl_pct=gross_pnl_pct,
            commission_pct=commission_pct,
            net_pnl_pct=net_pnl_pct,
            candle_time=candle_time,
            dry_run=True,
            exit_reason=exit_reason,
        )
        risk_state.daily_realized_pnl += ret_net
        risk_state.open_positions_total = max(0, risk_state.open_positions_total - 1)
        if sector and sector in risk_state.open_by_sector:
            risk_state.open_by_sector[sector] = max(
                0,
                risk_state.open_by_sector[sector] - 1,
            )
        state.in_position = False
        state.entry_price = Decimal("0")
        return

    from uuid import uuid4

    try:
        resp = client.orders.post_order(
            figi=figi,
            quantity=lots,
            direction=OrderDirection.ORDER_DIRECTION_SELL,
            account_id=account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
            order_id=str(uuid4()),
        )
        logging.info(
            "SELL отправлен: %s x%d, resp=%s, gross=%.2f%%, net=%.2f%%, reason=%s",
            ticker,
            lots,
            resp,
            gross_pnl_pct,
            net_pnl_pct,
            exit_reason,
        )
        log_trade(
            ticker=ticker,
            side="SELL",
            price=last_price,
            lots=lots,
            gross_pnl_pct=gross_pnl_pct,
            commission_pct=commission_pct,
            net_pnl_pct=net_pnl_pct,
            candle_time=candle_time,
            dry_run=False,
            exit_reason=exit_reason,
        )
        risk_state.daily_realized_pnl += ret_net
        risk_state.open_positions_total = max(0, risk_state.open_positions_total - 1)
        if sector and sector in risk_state.open_by_sector:
            risk_state.open_by_sector[sector] = max(
                0,
                risk_state.open_by_sector[sector] - 1,
            )
        state.in_position = False
        state.entry_price = Decimal("0")
    except InvestError as e:
        logging.error("Ошибка SELL по %s: %s", ticker, e)


# ================== ЛОГИКА СИГНАЛОВ ==================

def should_open_long(state: InstrumentState, volume: int, ret: float) -> bool:
    z_vol = calc_z_score(float(volume), state.vol_mean, state.vol_std)
    z_ret = calc_z_score(ret, state.ret_mean, state.ret_std)

    vol_thr = state.vol_z_entry or BASE_VOL_Z_ENTRY
    ret_thr = state.ret_z_entry or BASE_RET_Z_ENTRY

    # тренд-лонг: большой объём + сильный рост, с порогами под тикер
    return (z_vol >= vol_thr) and (z_ret >= ret_thr)


def should_close_long(
    state: InstrumentState,
    high_price: Decimal,
    low_price: Decimal,
) -> Tuple[bool, Optional[str]]:
    """
    Закрываем по касанию TP/SL на основе high/low текущей свечи.
    Приоритет SL (чтобы не оптимизировать в плюс, когда за одну свечу задели и SL, и TP).
    """
    if not state.in_position or state.entry_price == 0:
        return False, None

    entry = state.entry_price

    # ретёрны в долях для high/low
    ret_high = float((high_price / entry) - 1)
    ret_low = float((low_price / entry) - 1)

    tp = state.take_profit_pct  # типа 0.0025
    sl = state.stop_loss_pct    # типа -0.0015

    # Сначала смотрим SL (консервативно)
    if ret_low <= sl:
        return True, "SL_TOUCH"

    # Потом TP
    if ret_high >= tp:
        return True, "TP_TOUCH"

    return False, None


# ================== СИНК ПОЗИЦИЙ С БРОКЕРОМ ==================

def sync_positions_from_broker(
    client: Client,
    account_id: str,
    state_by_ticker: Dict[str, InstrumentState],
    figi_to_ticker: Dict[str, str],
):
    """
    При старте смотрим текущие позиции у брокера и помечаем соответствующие тикеры in_position=True.
    entry_price инициализируем по первой свече.
    """
    try:
        resp = client.operations.get_positions(account_id=account_id)
    except Exception as e:
        logging.error("Не удалось получить позиции для синка: %s", e)
        return

    logging.info("get_positions: всего позиций в securities=%d", len(resp.securities))

    synced = 0

    for sec in resp.securities:
        figi = getattr(sec, "figi", None)
        if not figi:
            continue

        ticker = figi_to_ticker.get(figi)
        if not ticker:
            continue

        state = state_by_ticker.get(ticker)
        if not state:
            continue

        qty = getattr(sec, "balance", getattr(sec, "quantity", None))

        state.in_position = True
        logging.info(
            "Синхронизирована позиция по %s (figi=%s, qty=%s) – помечаем in_position=True",
            ticker,
            figi,
            qty,
        )
        synced += 1

    logging.info("Синхронизировано позиций с брокером: %d тикеров", synced)


def rebuild_risk_state_from_positions(
    risk_state: RiskState,
    state_by_ticker: Dict[str, InstrumentState],
    ticker_to_sector: Dict[str, Optional[str]],
):
    """
    После синка позиций восстанавливаем open_positions_total и open_by_sector.
    """
    risk_state.open_positions_total = 0
    risk_state.open_by_sector = {}

    for ticker, st in state_by_ticker.items():
        if not st.in_position:
            continue
        risk_state.open_positions_total += 1
        sector = ticker_to_sector.get(ticker)
        if sector:
            risk_state.open_by_sector[sector] = risk_state.open_by_sector.get(sector, 0) + 1

    logging.info(
        "После синка: всего открытых позиций=%d",
        risk_state.open_positions_total,
    )


# ================== ОБРАБОТКА СВЕЧИ ==================

def process_candle(
    client: Client,
    account_id: str,
    ticker: str,
    figi: str,
    state: InstrumentState,
    candle,
    sector: Optional[str],
    risk_state: RiskState,
):
    close_price = quotation_to_decimal(candle.close)
    high_price  = quotation_to_decimal(candle.high)
    low_price   = quotation_to_decimal(candle.low)
    volume = candle.volume

    if state.last_price == 0:
        ret = 0.0
    else:
        ret = float((close_price / state.last_price) - 1)

    state.last_price = close_price

    # Если поза уже была помечена как открытая, но entry_price ещё не знаем (после синка)
    if state.in_position and state.entry_price == 0:
        state.entry_price = close_price
        logging.info(
            "[%s] Инициализируем entry_price после синка: %s",
            ticker,
            close_price,
        )

    logging.info(
        "[%s] %s close=%s high=%s low=%s vol=%s in_pos=%s",
        ticker,
        candle.time,
        close_price,
        high_price,
        low_price,
        volume,
        state.in_position,
    )

    if not state.in_position:
        if not can_open_new_position(ticker, sector, risk_state):
            return
        # Сначала проверяем аномалию по объёму/ретёрну
        if not should_open_long(state, volume, ret):
            return

        # Теперь проверяем спред через стакан
        spread_pct = get_current_spread_pct(client, figi)
        if spread_pct is None:
            logging.info(
                "[%s] Не удалось получить спред (нет стакана или ошибка), пропускаем сигнал.",
                ticker,
            )
            return

        if spread_pct > state.max_spread_pct:
            logging.info(
                "[%s] Спред %.4f%% выше лимита %.4f%%, пропускаем сигнал.",
                ticker,
                spread_pct * 100,
                state.max_spread_pct * 100,
            )
            return

        # Всё ок: и сигнал, и спред
        open_long(client, account_id, figi, ticker, state, candle.time, risk_state, sector)
    else:
        # Здесь TP/SL проверяется на КАЖДОМ обновлении свечи по high/low (по касанию)
        should_exit, exit_reason = should_close_long(state, high_price, low_price)
        if should_exit and exit_reason:
            close_long(
                client,
                account_id,
                figi,
                ticker,
                state,
                candle.time,
                risk_state,
                sector,
                exit_reason=exit_reason,
            )


# ================== MAIN ==================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    token = os.environ.get(TOKEN_ENV_NAME)
    if not token:
        raise RuntimeError(
            f"Переменная окружения {TOKEN_ENV_NAME} не установлена. "
            f'Сделай:  $env:{TOKEN_ENV_NAME}="ТВОЙ_ТОКЕН"'
        )

    logging.info("Запуск многотикерного робота. DRY_RUN=%s", DRY_RUN)
    logging.info("Загружаем историю из %s ...", HISTORY_DIR)
    states_from_history = build_states_from_history()
    if not states_from_history:
        logging.error("Нет валидных тикеров в истории. Сначала запусти сбор истории (1m).")
        return

    # ---- грузим / создаём config.json и накрываем им стейты ----
    config = load_or_init_config(CONFIG_FILE)
    config = ensure_config_has_all_tickers(config, states_from_history, CONFIG_FILE)
    apply_config_to_states(states_from_history, config)

    risk_state = RiskState()

    with Client(token) as client:
        accounts = client.users.get_accounts()
        if not accounts.accounts:
            raise RuntimeError("Не найден ни один счёт в Тинькофф Инвестициях")

        account_id_env = os.environ.get(ACCOUNT_ID_ENV_NAME)

        if account_id_env:
            # Проверяем, что такой счёт реально есть
            account_obj = None
            for acc in accounts.accounts:
                if acc.id == account_id_env:
                    account_obj = acc
                    break
            if account_obj is None:
                logging.error(
                    "ACCOUNT_ID из %s=%s не найден среди счетов.",
                    ACCOUNT_ID_ENV_NAME,
                    account_id_env,
                )
                logging.info("Доступные счета:")
                for acc in accounts.accounts:
                    logging.info("  id=%s, type=%s, name=%s", acc.id, acc.type, acc.name)
                raise RuntimeError(
                    f"Неверный {ACCOUNT_ID_ENV_NAME}. Скопируй корректный id из логов и задай заново."
                )
            else:
                account_id = account_id_env
                logging.info(
                    "Используем account_id из переменной окружения: %s (type=%s, name=%s)",
                    account_id,
                    account_obj.type,
                    account_obj.name,
                )
        else:
            logging.info("Переменная %s не задана. Список доступных счетов:", ACCOUNT_ID_ENV_NAME)
            for acc in accounts.accounts:
                logging.info("  id=%s, type=%s, name=%s", acc.id, acc.type, acc.name)
            raise RuntimeError(
                f"Выбери нужный id из логов и задай переменную окружения {ACCOUNT_ID_ENV_NAME} "
                f'например:  $env:{ACCOUNT_ID_ENV_NAME}="ВСТАВЬ_ID"  и перезапусти робота.'
            )

        # Получаем RUB-акции у брокера
        ticker_to_figi, ticker_to_sector = get_moex_shares_meta(client)

        # Пересечение тикеров: только те, по которым есть история и есть у брокера
        history_tickers = set(states_from_history.keys())
        broker_tickers = set(ticker_to_figi.keys())
        common_tickers = sorted(history_tickers & broker_tickers)

        logging.info("Тикеров в истории: %d", len(history_tickers))
        logging.info("Тикеров у брокера (RUB): %d", len(broker_tickers))
        logging.info("Общих тикеров (история ∩ брокер): %d", len(common_tickers))

        if not common_tickers:
            logging.error("Нет пересечения тикеров между историей и брокером")
            return

        if MAX_INSTRUMENTS is not None and len(common_tickers) > MAX_INSTRUMENTS:
            logging.info(
                "Всего общих тикеров %d, ограничиваемся первыми %d",
                len(common_tickers),
                MAX_INSTRUMENTS,
            )
            common_tickers = common_tickers[:MAX_INSTRUMENTS]

        instruments = []
        state_by_ticker: Dict[str, InstrumentState] = {}
        figi_to_ticker: Dict[str, str] = {}

        subscribed_tickers = []

        for ticker in common_tickers:
            # Жёстко вырезаем нежелательные тикеры
            if ticker in EXCLUDED_TICKERS:
                logging.info("Тикер %s в списке исключений, пропускаем", ticker)
                continue

            st = states_from_history[ticker]
            if not st.enabled:
                logging.info("Тикер %s отключен в конфиге (enabled=false), пропускаем", ticker)
                continue

            figi = ticker_to_figi[ticker]
            instruments.append(
                CandleInstrument(
                    figi=figi,
                    interval=SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE,
                )
            )
            state_by_ticker[ticker] = st
            figi_to_ticker[figi] = ticker
            subscribed_tickers.append(ticker)

        if not instruments:
            logging.error("Нет ни одного тикера для подписки (все отключены или исключены?).")
            return

        logging.info(
            "Подписываемся на свечи по тикерам (%d): %s",
            len(subscribed_tickers),
            ", ".join(subscribed_tickers),
        )

        # ---- Синхронизируем уже открытые позиции с брокером ----
        sync_positions_from_broker(
            client=client,
            account_id=account_id,
            state_by_ticker=state_by_ticker,
            figi_to_ticker=figi_to_ticker,
        )
        rebuild_risk_state_from_positions(
            risk_state=risk_state,
            state_by_ticker=state_by_ticker,
            ticker_to_sector=ticker_to_sector,
        )

        market_data_stream: MarketDataStreamManager = client.create_market_data_stream()
        # ВАЖНО: без waiting_close(), чтобы работать внутри свечи
        market_data_stream.candles.subscribe(instruments)

        try:
            for marketdata in market_data_stream:
                if marketdata.candle is None:
                    continue

                candle = marketdata.candle
                figi = candle.figi
                ticker = figi_to_ticker.get(figi)
                if ticker is None:
                    continue

                state = state_by_ticker[ticker]
                sector = ticker_to_sector.get(ticker)
                process_candle(
                    client,
                    account_id,
                    ticker,
                    figi,
                    state,
                    candle,
                    sector,
                    risk_state,
                )

        except KeyboardInterrupt:
            logging.info("Остановлено с клавиатуры, выходим…")
        except Exception as e:
            logging.exception("Необработанная ошибка: %s", e)


if __name__ == "__main__":
    main()
