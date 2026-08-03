# Places orders in a locally-running MetaTrader 5 terminal via MetaQuotes'
# official `MetaTrader5` Python package. That package talks to the terminal
# over a local IPC connection, not a network API - it only works if MT5 is
# installed and logged in on the SAME machine this script runs on, and (per
# MetaQuotes) only ships Windows builds, so this needs a Windows PC or VPS.
#
# DRY_RUN defaults to True in config.example.py on purpose: verify parsed
# signals look right (test_parser.py, then real listener.py output) before
# ever flipping it off and risking real money on an unverified parser.

import MetaTrader5 as mt5


def connect():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")


def shutdown():
    mt5.shutdown()


def calc_lot_size(symbol: str, sl_distance_price: float, risk_amount: float) -> float:
    """Position size such that hitting the stop loses ~risk_amount, in the
    account's currency. Same risk-based sizing model used throughout this
    project's backtests, adapted to MT5's tick_value/tick_size (accounts
    for non-USD-quoted instruments and indices, unlike a flat pip-value
    assumption)."""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"Unknown symbol {symbol!r} - check it's visible in MT5's Market Watch")
    if sl_distance_price <= 0 or info.trade_tick_size <= 0:
        return info.volume_min

    value_per_price_unit_per_lot = info.trade_tick_value / info.trade_tick_size
    risk_per_lot = sl_distance_price * value_per_price_unit_per_lot
    if risk_per_lot <= 0:
        return info.volume_min

    lots = risk_amount / risk_per_lot
    lots = max(info.volume_min, min(lots, info.volume_max))
    step = info.volume_step
    lots = round(lots / step) * step
    return round(lots, 2)


def place_market_order(symbol, side, lots, sl=None, tp=None, deviation=20,
                        magic=990011, comment="tg-copier", dry_run=True):
    if not mt5.symbol_select(symbol, True):
        raise ValueError(f"Could not select symbol {symbol!r} in Market Watch")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise ValueError(f"No tick data for {symbol!r}")

    price = tick.ask if side == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl or 0.0,
        "tp": tp or 0.0,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if dry_run:
        print(f"[DRY RUN] would send: {request}")
        return None

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send returned None: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"order_send failed: retcode={result.retcode} comment={result.comment}")
    return result
