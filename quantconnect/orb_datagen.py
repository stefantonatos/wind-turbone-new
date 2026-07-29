# Generates labeled training data for an ML classifier that scores ORB
# breakouts, from quantconnect/orb.py's rules. This is NOT a trading
# algorithm to run live/paper - its only job is to walk historical data,
# take EVERY close-confirmed breakout (none of orb.py's hand-coded filters
# are applied as gates here - only computed as features), and record what
# happened, so train_orb_model.ipynb has real examples to learn "which
# breakouts actually work" from instead of my hand-picked thresholds.
#
# Deliberately different from orb.py in two ways, both about maximizing
# training examples rather than mirroring live-trading discipline:
#   - No hand-coded filters block entry (RANGE_ATR_FILTER, TREND_FILTER,
#     IMPULSE_FILTER, VOLUME_FILTER, VOLATILITY_REGIME_FILTER in orb.py) -
#     every close-confirmed breakout is taken and labeled.
#   - More than one trade per day is allowed (still only one AT A TIME -
#     Portfolio/GetOpenOrders gate, same as elsewhere in this project) so a
#     day with multiple genuine breakouts contributes more than one row.
#
# Output: a CSV written to QuantConnect's Object Store (key
# "orb_training_data") - NOT to self.Debug, which has a 10KB-per-backtest
# cap this project already got badly burned by once. Download it from the
# project's Object Store / Data panel in the QC UI after the backtest
# finishes, then run train_orb_model.ipynb against it.
#
# IMPORTANT: the feature computation here (UpdateSMMA/UpdateATR and the
# features dict in OnFiveMinuteBar) must stay IDENTICAL to orb_ml.py's -
# a trained model is only as good as getting the exact same numbers at
# inference time that it saw at training time. Both files are
# self-contained (no shared import) since QC project file-sharing on
# mobile is unreliable - if you change the feature logic in one, change it
# in the other, in the same way.

from AlgorithmImports import *
from datetime import time, date
import csv
import io


FEATURE_COLUMNS = [
    "range_size_pips", "range_vs_atr", "impulse_ratio", "atr_regime_ratio",
    "trend_strength", "volume_ratio", "hour_of_day", "minute_of_hour",
    "day_of_week", "direction",
]


class ORBDataGenerator(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2025, 1, 1)   # 5 years - raw breakouts are rare enough per day that 1 year alone is thin
        self.SetCash(10000)
        self.SetTimeZone(TimeZones.Utc)

        self.symbol = self.AddForex("EURUSD", Resolution.Minute, Market.Oanda).Symbol

        # --- same constants as orb.py (structural, not filters) ---
        self.SESSION_START = time(8, 0)
        self.RANGE_MINUTES = 15
        self.ENTRY_WINDOW_MINUTES = 180
        self.SESSION_END = time(21, 0)
        self.ENTRY_BUFFER_PIPS = 1.5
        self.REWARD_RISK = 1.0
        self.RISK_PERCENT = 0.01
        self.MIN_SL_PIPS = 3
        self.MAX_LEVERAGE = 10
        self.PIP = 0.0001
        self.TREND_MA_LEN = 200
        self.ATR_LEN = 14
        self.RANGE_AVG_LEN = 20
        self.VOLUME_AVG_LEN = 20
        self.ATR_BASELINE_LEN = 100

        # --- persisted indicators ---
        self.ma_val = None
        self.ma_seed = []
        self.atr_val = None
        self.atr_seed = []
        self.prev_close = None
        self.range_avg_val = None
        self.range_avg_seed = []
        self.volume_avg_val = None
        self.volume_avg_seed = []
        self.atr_baseline_val = None
        self.atr_baseline_seed = []

        # --- per-day state ---
        self.current_day = None
        self.range_high = None
        self.range_low = None

        # --- open-trade bookkeeping for labeling ---
        self.entry_ticket = None
        self.sl_ticket = None
        self.tp_ticket = None
        self.current_features = None   # dict snapshot taken at entry
        self.awaiting_liquidation = False

        self.training_rows = []

        consolidator = QuoteBarConsolidator(timedelta(minutes=5))
        consolidator.DataConsolidated += self.OnFiveMinuteBar
        self.SubscriptionManager.AddConsolidator(self.symbol, consolidator)

    # --- indicator math: identical to orb.py/orb_ml.py ---

    def UpdateSMMA(self, current, seed_buf, new_value, length):
        if current is None:
            seed_buf.append(new_value)
            if len(seed_buf) < length:
                return None, seed_buf
            return sum(seed_buf) / length, []
        return (current * (length - 1) + new_value) / length, seed_buf

    def UpdateATR(self, high, low, close, length):
        if self.prev_close is not None:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
            if self.atr_val is None:
                self.atr_seed.append(tr)
                if len(self.atr_seed) >= length:
                    self.atr_val = sum(self.atr_seed) / length
                    self.atr_seed = []
            else:
                self.atr_val = (self.atr_val * (length - 1) + tr) / length
        self.prev_close = close
        return self.atr_val

    def RangeEndTime(self):
        base = datetime.combine(date.min, self.SESSION_START)
        return (base + timedelta(minutes=self.RANGE_MINUTES)).time()

    def EntryEndTime(self):
        base = datetime.combine(date.min, self.SESSION_START)
        return (base + timedelta(minutes=self.RANGE_MINUTES + self.ENTRY_WINDOW_MINUTES)).time()

    def _record_outcome(self, label, exit_price, reason):
        if self.current_features is None:
            return
        row = dict(self.current_features)
        row["label"] = label
        row["exit_reason"] = reason
        self.training_rows.append(row)
        self.current_features = None

    def OnOrderEvent(self, order_event):
        if order_event.Status != OrderStatus.Filled:
            return

        if self.sl_ticket is not None and order_event.OrderId == self.sl_ticket.OrderId:
            self._record_outcome(0, order_event.FillPrice, "SL")
            if self.tp_ticket is not None:
                self.tp_ticket.Cancel()
            self.entry_ticket = self.sl_ticket = self.tp_ticket = None

        elif self.tp_ticket is not None and order_event.OrderId == self.tp_ticket.OrderId:
            self._record_outcome(1, order_event.FillPrice, "TP")
            if self.sl_ticket is not None:
                self.sl_ticket.Cancel()
            self.entry_ticket = self.sl_ticket = self.tp_ticket = None

        elif self.entry_ticket is not None and order_event.OrderId == self.entry_ticket.OrderId:
            pass  # entry fill - features already snapshotted at signal time, nothing to record yet

        elif self.awaiting_liquidation and self.current_features is not None:
            exit_price = order_event.FillPrice
            entry_price = self.current_features["_entry_price"]
            side = self.current_features["_side"]
            pnl = (exit_price - entry_price) if side == "BUY" else (entry_price - exit_price)
            self._record_outcome(1 if pnl > 0 else 0, exit_price, "FLAT")
            for ticket in (self.sl_ticket, self.tp_ticket):
                if ticket is not None:
                    ticket.Cancel()
            self.entry_ticket = self.sl_ticket = self.tp_ticket = None
            self.awaiting_liquidation = False

    def OnFiveMinuteBar(self, sender, bar):
        t = self.Time
        today = t.date()
        tod = t.time()

        if today != self.current_day:
            self.current_day = today
            self.range_high = None
            self.range_low = None

        self.ma_val, self.ma_seed = self.UpdateSMMA(self.ma_val, self.ma_seed, bar.Close, self.TREND_MA_LEN)
        current_atr = self.UpdateATR(bar.High, bar.Low, bar.Close, self.ATR_LEN)
        self.range_avg_val, self.range_avg_seed = self.UpdateSMMA(
            self.range_avg_val, self.range_avg_seed, bar.High - bar.Low, self.RANGE_AVG_LEN)
        bar_volume = getattr(bar, "Volume", 0) or 0
        self.volume_avg_val, self.volume_avg_seed = self.UpdateSMMA(
            self.volume_avg_val, self.volume_avg_seed, bar_volume, self.VOLUME_AVG_LEN)
        if current_atr is not None:
            self.atr_baseline_val, self.atr_baseline_seed = self.UpdateSMMA(
                self.atr_baseline_val, self.atr_baseline_seed, current_atr, self.ATR_BASELINE_LEN)

        # Flatten at session end regardless of anything else.
        if tod >= self.SESSION_END:
            if self.Portfolio[self.symbol].Quantity != 0:
                self.awaiting_liquidation = True
                self.Liquidate(self.symbol)
            return

        if self.Portfolio[self.symbol].Quantity != 0 or len(self.Transactions.GetOpenOrders(self.symbol)) > 0:
            return

        range_end = self.RangeEndTime()
        entry_end = self.EntryEndTime()

        if self.SESSION_START <= tod < range_end:
            self.range_high = bar.High if self.range_high is None else max(self.range_high, bar.High)
            self.range_low = bar.Low if self.range_low is None else min(self.range_low, bar.Low)
            return

        if self.range_high is None or tod >= entry_end:
            return

        buffer_price = self.ENTRY_BUFFER_PIPS * self.PIP
        price = bar.Close
        buy_setup = price > self.range_high + buffer_price
        sell_setup = price < self.range_low - buffer_price
        if not buy_setup and not sell_setup:
            return

        range_size = self.range_high - self.range_low
        bar_range = bar.High - bar.Low

        features = {
            "range_size_pips": range_size / self.PIP,
            "range_vs_atr": (range_size / current_atr) if current_atr else "",
            "impulse_ratio": (bar_range / self.range_avg_val) if self.range_avg_val else "",
            "atr_regime_ratio": (current_atr / self.atr_baseline_val) if (current_atr and self.atr_baseline_val) else "",
            "trend_strength": ((price - self.ma_val) / current_atr) if (self.ma_val is not None and current_atr) else "",
            "volume_ratio": (bar_volume / self.volume_avg_val) if (self.volume_avg_val and self.volume_avg_val > 0) else "",
            "hour_of_day": tod.hour,
            "minute_of_hour": tod.minute,
            "day_of_week": t.weekday(),
            "direction": 1 if buy_setup else -1,
            "_entry_price": price,
            "_side": "BUY" if buy_setup else "SELL",
        }

        sl_distance = max(range_size, self.MIN_SL_PIPS * self.PIP)
        tp_distance = sl_distance * self.REWARD_RISK
        equity = self.Portfolio.TotalPortfolioValue
        risk_amount = equity * self.RISK_PERCENT
        quantity = min(risk_amount / sl_distance, (equity * self.MAX_LEVERAGE) / price)

        self.current_features = features

        if buy_setup:
            self.entry_ticket = self.MarketOrder(self.symbol, quantity)
            self.sl_ticket = self.StopMarketOrder(self.symbol, -quantity, price - sl_distance)
            self.tp_ticket = self.LimitOrder(self.symbol, -quantity, price + tp_distance)
        else:
            self.entry_ticket = self.MarketOrder(self.symbol, -quantity)
            self.sl_ticket = self.StopMarketOrder(self.symbol, quantity, price + sl_distance)
            self.tp_ticket = self.LimitOrder(self.symbol, quantity, price - tp_distance)

    def OnEndOfAlgorithm(self):
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=FEATURE_COLUMNS + ["label", "exit_reason"])
        writer.writeheader()
        for row in self.training_rows:
            writer.writerow({k: row.get(k, "") for k in FEATURE_COLUMNS + ["label", "exit_reason"]})
        self.ObjectStore.Save("orb_training_data", buf.getvalue())
        self.Debug(f"Wrote {len(self.training_rows)} labeled rows to Object Store key 'orb_training_data'")
