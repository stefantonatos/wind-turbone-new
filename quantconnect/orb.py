# QuantConnect (LEAN) backtest of an Opening Range Breakout (ORB) strategy
# on EUR/USD, built from research into what actually holds up rather than
# the textbook version: independent backtests (QuantifiedStrategies' London
# Breakout study) found the naive "breakout + hold to a time exit" version
# loses money on EUR/USD, consistent with what this project already found
# for the taught trend strategy and for a plain Donchian breakout
# (main.py, donchian.py) - and one source found FADING the open range
# showed more promise, echoing this project's own earlier "reverse it"
# result. So: build it properly, with the filters research says matter,
# and let REVERSE_SIGNALS make straight-vs-fade a one-line comparison
# rather than a guess.
#
# Rules (see README.md for the research this is based on):
#   - Opening range = high/low of the first RANGE_MINUTES after SESSION_START
#     (default: 08:00 UTC, the London open - matches this project's live
#     bot's trading window).
#   - Entry requires a bar CLOSE beyond the range (+ a small buffer), not a
#     bare touch - literature flags close-confirmation as reducing false
#     breakouts vs. resting stop orders sitting right at the level.
#   - SL = the opposite side of the opening range (the classic ORB stop),
#     floored so a very tight range can't blow up position size. TP = SL
#     distance x REWARD_RISK; REWARD_RISK=1.0 matches the textbook "Target 1"
#     measured-move target (range height projected from the breakout).
#   - Optional TREND_FILTER: only take the breakout in the direction of a
#     persisted 200-bar 5-min MA (the "higher timeframe trend" filter
#     research flagged as one of the more credible improvements).
#   - Optional RANGE_ATR_FILTER: skip if the range is too small relative to
#     ATR(14) (likely noise, breakout won't have room to run) or too large
#     (move may already be exhausted).
#   - One trade per day. Entry window closes ENTRY_WINDOW_MINUTES after the
#     range forms - if no breakout by then, no trade that day. Any open
#     position is flattened at SESSION_END (day-trade only, no overnight
#     swap/gap risk).
#
# Execution follows this project's already-verified-safe patterns from
# main.py/donchian.py: persisted (not re-windowed) indicators, real
# StopMarketOrder/LimitOrder brackets, an OnOrderEvent OCO handler, and
# Portfolio/GetOpenOrders as the sole "are we exposed" gate - no custom
# flags to get wrong.

from AlgorithmImports import *
from datetime import time, date  # AlgorithmImports re-exports datetime/timedelta but not these explicitly


class OpeningRangeBreakoutStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(10000)
        self.SetTimeZone(TimeZones.Utc)  # unambiguous session times; QC's default is America/New_York

        self.symbol = self.AddForex("EURUSD", Resolution.Minute, Market.Oanda).Symbol

        # --- config ---
        self.REVERSE_SIGNALS = False   # flip to True to fade the breakout instead of taking it
        self.SESSION_START = time(8, 0)     # London open, UTC
        self.RANGE_MINUTES = 15
        self.ENTRY_WINDOW_MINUTES = 180     # breakout must confirm within 3h of the range forming
        self.SESSION_END = time(21, 0)      # flatten by here, UTC - no overnight carry
        self.ENTRY_BUFFER_PIPS = 1.5        # close must clear the range by this much (reduces false breaks)
        self.REWARD_RISK = 1.0              # TP distance = SL distance x this. 1.0 = classic measured-move target
        self.RISK_PERCENT = 0.01
        self.MIN_SL_PIPS = 3
        self.MAX_LEVERAGE = 10
        self.PIP = 0.0001
        self.TREND_FILTER = True
        self.TREND_MA_LEN = 200
        self.RANGE_ATR_FILTER = True
        self.ATR_LEN = 14
        self.MIN_RANGE_ATR_MULT = 0.5
        self.MAX_RANGE_ATR_MULT = 3.0

        # --- persisted indicators (updated every bar, never re-windowed -
        # see main.py/donchian.py's comments on why a trimmed rolling
        # buffer silently corrupts a long MA/ATR) ---
        self.ma_val = None
        self.ma_seed = []
        self.atr_val = None
        self.atr_seed = []
        self.prev_close = None

        # --- per-day state ---
        self.current_day = None
        self.range_high = None
        self.range_low = None
        self.traded_today = False

        # --- order tickets ---
        self.sl_ticket = None
        self.tp_ticket = None

        consolidator = QuoteBarConsolidator(timedelta(minutes=5))
        consolidator.DataConsolidated += self.OnFiveMinuteBar
        self.SubscriptionManager.AddConsolidator(self.symbol, consolidator)

    # --- indicator math: same verified incremental patterns as main.py/donchian.py ---

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

    # Cancel the sibling SL/TP order once one of them fills, so it doesn't
    # sit resting against a position that no longer exists.
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

    # --- main bar handler ---

    def OnFiveMinuteBar(self, sender, bar):
        t = self.Time
        today = t.date()
        tod = t.time()

        if today != self.current_day:
            self.current_day = today
            self.range_high = None
            self.range_low = None
            self.traded_today = False

        # Persisted indicators, updated every bar unconditionally.
        self.ma_val, self.ma_seed = self.UpdateSMMA(self.ma_val, self.ma_seed, bar.Close, self.TREND_MA_LEN)
        current_atr = self.UpdateATR(bar.High, bar.Low, bar.Close, self.ATR_LEN)

        # Flatten at session end regardless of anything else - day-trade
        # only, no overnight swap/gap risk.
        if tod >= self.SESSION_END:
            if self.Portfolio[self.symbol].Quantity != 0:
                self.Liquidate(self.symbol)
            for ticket in (self.sl_ticket, self.tp_ticket):
                if ticket is not None:
                    ticket.Cancel()
            self.sl_ticket = None
            self.tp_ticket = None
            return

        # Single source of truth for "are we already exposed": actual
        # position size, or a bracket order still resting.
        if self.Portfolio[self.symbol].Quantity != 0 or len(self.Transactions.GetOpenOrders(self.symbol)) > 0:
            return

        range_end = self.RangeEndTime()
        entry_end = self.EntryEndTime()

        if self.SESSION_START <= tod < range_end:
            self.range_high = bar.High if self.range_high is None else max(self.range_high, bar.High)
            self.range_low = bar.Low if self.range_low is None else min(self.range_low, bar.Low)
            return

        if self.range_high is None or self.traded_today:
            return  # range not formed yet today, or already traded/skipped today

        if tod >= entry_end:
            self.traded_today = True  # entry window expired with no breakout
            return

        buffer_price = self.ENTRY_BUFFER_PIPS * self.PIP
        price = bar.Close
        buy_setup = price > self.range_high + buffer_price
        sell_setup = price < self.range_low - buffer_price

        if self.REVERSE_SIGNALS:
            buy_setup, sell_setup = sell_setup, buy_setup

        if not buy_setup and not sell_setup:
            return

        range_size = self.range_high - self.range_low

        if self.RANGE_ATR_FILTER and current_atr:
            if range_size < self.MIN_RANGE_ATR_MULT * current_atr or range_size > self.MAX_RANGE_ATR_MULT * current_atr:
                self.traded_today = True
                return

        if self.TREND_FILTER and self.ma_val is not None:
            if buy_setup and not (price > self.ma_val):
                self.traded_today = True
                return
            if sell_setup and not (price < self.ma_val):
                self.traded_today = True
                return

        sl_distance = max(range_size, self.MIN_SL_PIPS * self.PIP)
        tp_distance = sl_distance * self.REWARD_RISK

        equity = self.Portfolio.TotalPortfolioValue
        risk_amount = equity * self.RISK_PERCENT
        quantity = risk_amount / sl_distance
        max_quantity = (equity * self.MAX_LEVERAGE) / price
        quantity = min(quantity, max_quantity)

        self.traded_today = True  # one trade per day, win or lose

        self.Debug(
            f"{self.Time} ENTRY {'BUY' if buy_setup else 'SELL'} range=[{self.range_low:.5f},{self.range_high:.5f}] "
            f"equity={equity:.2f} qty={quantity:.0f} sl_dist={sl_distance:.5f} risk$={risk_amount:.2f}"
        )

        if buy_setup:
            self.MarketOrder(self.symbol, quantity)
            self.sl_ticket = self.StopMarketOrder(self.symbol, -quantity, price - sl_distance)
            self.tp_ticket = self.LimitOrder(self.symbol, -quantity, price + tp_distance)
        elif sell_setup:
            self.MarketOrder(self.symbol, -quantity)
            self.sl_ticket = self.StopMarketOrder(self.symbol, quantity, price + sl_distance)
            self.tp_ticket = self.LimitOrder(self.symbol, quantity, price - tp_distance)
