# Live/backtest ORB algorithm using the classifier trained by
# train_orb_model.ipynb, instead of (or in addition to) orb.py's
# hand-coded filters. Every close-confirmed breakout's features are
# computed exactly as in orb_datagen.py, fed to the trained model's
# predict_proba(), and the trade is only taken if confidence clears
# ML_CONFIDENCE_THRESHOLD - pick that threshold from
# train_orb_model.ipynb's R-multiple sweep on its held-out test set, not a
# guess.
#
# Requires: run orb_datagen.py as a backtest in THIS SAME QuantConnect
# project, then run train_orb_model.ipynb in this project's Research tab.
# Object Store is project-scoped, so the "orb_ml_model" key this loads
# below only exists if both of those happened first, in this project.
#
# IMPORTANT: the feature computation here (UpdateSMMA/UpdateATR and the
# features dict in OnFiveMinuteBar) must stay IDENTICAL to
# orb_datagen.py's - see that file's header comment. This file is
# self-contained by design (no shared import, for QC mobile-project-file
# reliability), so keep the two in sync by hand.

from AlgorithmImports import *
from datetime import time, date
import pickle


FEATURE_COLUMNS = [
    "range_size_pips", "range_vs_atr", "impulse_ratio", "atr_regime_ratio",
    "trend_strength", "volume_ratio", "hour_of_day", "minute_of_hour",
    "day_of_week", "direction",
]


class ORBMLStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(10000)
        self.SetTimeZone(TimeZones.Utc)

        self.symbol = self.AddForex("EURUSD", Resolution.Minute, Market.Oanda).Symbol

        # --- config ---
        self.SESSION_START = time(8, 0)
        self.RANGE_MINUTES = 15
        self.ENTRY_WINDOW_MINUTES = 180
        self.SESSION_END = time(21, 0)
        self.ENTRY_BUFFER_PIPS = 1.5
        self.REWARD_RISK = 1.0        # must match what train_orb_model.ipynb assumed for its R-multiple sweep
        self.RISK_PERCENT = 0.01
        self.MIN_SL_PIPS = 3
        self.MAX_LEVERAGE = 10
        self.PIP = 0.0001
        self.TREND_MA_LEN = 200
        self.ATR_LEN = 14
        self.RANGE_AVG_LEN = 20
        self.VOLUME_AVG_LEN = 20
        self.ATR_BASELINE_LEN = 100
        self.ML_CONFIDENCE_THRESHOLD = 0.60   # pick from the notebook's R-multiple-vs-threshold sweep

        # --- load the trained model ---
        self.model = None
        if self.ObjectStore.ContainsKey("orb_ml_model"):
            model_bytes = self.ObjectStore.ReadBytes("orb_ml_model")
            self.model = pickle.loads(bytes(model_bytes))
            self.Debug("Loaded orb_ml_model from Object Store")
        else:
            self.Debug(
                "WARNING: no 'orb_ml_model' found in Object Store - run orb_datagen.py then "
                "train_orb_model.ipynb in this same project first. No trades will be taken."
            )

        # --- persisted indicators (identical to orb_datagen.py) ---
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
        self.traded_today = False   # unlike orb_datagen.py, this IS live-trading discipline: one trade/day

        # --- order tickets ---
        self.sl_ticket = None
        self.tp_ticket = None

        consolidator = QuoteBarConsolidator(timedelta(minutes=5))
        consolidator.DataConsolidated += self.OnFiveMinuteBar
        self.SubscriptionManager.AddConsolidator(self.symbol, consolidator)

    # --- indicator math: identical to orb_datagen.py ---

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

    def OnOrderEvent(self, order_event):
        if order_event.Status != OrderStatus.Filled:
            return
        if self.sl_ticket is not None and order_event.OrderId == self.sl_ticket.OrderId:
            if self.tp_ticket is not None:
                self.tp_ticket.Cancel()
            self.sl_ticket = None
            self.tp_ticket = None
        elif self.tp_ticket is not None and order_event.OrderId == self.tp_ticket.OrderId:
            if self.sl_ticket is not None:
                self.sl_ticket.Cancel()
            self.sl_ticket = None
            self.tp_ticket = None

    def OnFiveMinuteBar(self, sender, bar):
        if self.model is None:
            return

        t = self.Time
        today = t.date()
        tod = t.time()

        if today != self.current_day:
            self.current_day = today
            self.range_high = None
            self.range_low = None
            self.traded_today = False

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

        if tod >= self.SESSION_END:
            if self.Portfolio[self.symbol].Quantity != 0:
                self.Liquidate(self.symbol)
            for ticket in (self.sl_ticket, self.tp_ticket):
                if ticket is not None:
                    ticket.Cancel()
            self.sl_ticket = None
            self.tp_ticket = None
            return

        if self.Portfolio[self.symbol].Quantity != 0 or len(self.Transactions.GetOpenOrders(self.symbol)) > 0:
            return

        range_end = self.RangeEndTime()
        entry_end = self.EntryEndTime()

        if self.SESSION_START <= tod < range_end:
            self.range_high = bar.High if self.range_high is None else max(self.range_high, bar.High)
            self.range_low = bar.Low if self.range_low is None else min(self.range_low, bar.Low)
            return

        if self.range_high is None or self.traded_today:
            return

        if tod >= entry_end:
            self.traded_today = True
            return

        buffer_price = self.ENTRY_BUFFER_PIPS * self.PIP
        price = bar.Close
        buy_setup = price > self.range_high + buffer_price
        sell_setup = price < self.range_low - buffer_price
        if not buy_setup and not sell_setup:
            return

        range_size = self.range_high - self.range_low
        bar_range = bar.High - bar.Low

        # Same feature computation as orb_datagen.py, same handling of
        # "not enough history yet" (empty string there, None here - model
        # can't take a trade it doesn't have full features for).
        if current_atr is None or self.range_avg_val is None or self.atr_baseline_val is None or self.ma_val is None:
            self.traded_today = True
            return

        volume_ratio = (bar_volume / self.volume_avg_val) if (self.volume_avg_val and self.volume_avg_val > 0) else 1.0

        feature_row = [[
            range_size / self.PIP,
            range_size / current_atr,
            bar_range / self.range_avg_val,
            current_atr / self.atr_baseline_val,
            (price - self.ma_val) / current_atr,
            volume_ratio,
            tod.hour,
            tod.minute,
            t.weekday(),
            1 if buy_setup else -1,
        ]]

        confidence = self.model.predict_proba(feature_row)[0][1]

        if confidence < self.ML_CONFIDENCE_THRESHOLD:
            self.traded_today = True
            return

        sl_distance = max(range_size, self.MIN_SL_PIPS * self.PIP)
        tp_distance = sl_distance * self.REWARD_RISK
        equity = self.Portfolio.TotalPortfolioValue
        risk_amount = equity * self.RISK_PERCENT
        quantity = min(risk_amount / sl_distance, (equity * self.MAX_LEVERAGE) / price)

        self.traded_today = True

        self.Debug(
            f"{self.Time} ML-ENTRY {'BUY' if buy_setup else 'SELL'} confidence={confidence:.3f} "
            f"equity={equity:.2f} qty={quantity:.0f} sl_dist={sl_distance:.5f}"
        )

        if buy_setup:
            self.MarketOrder(self.symbol, quantity)
            self.sl_ticket = self.StopMarketOrder(self.symbol, -quantity, price - sl_distance)
            self.tp_ticket = self.LimitOrder(self.symbol, -quantity, price + tp_distance)
        else:
            self.MarketOrder(self.symbol, -quantity)
            self.sl_ticket = self.StopMarketOrder(self.symbol, quantity, price + sl_distance)
            self.tp_ticket = self.LimitOrder(self.symbol, quantity, price - tp_distance)
