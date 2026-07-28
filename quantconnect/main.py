# QuantConnect (LEAN) backtest of the same strategy as
# telegram-relay/src/strategy.js - trend-aligned 3 Line Strike / Engulfing
# arrow + RSI vs 50, gated by a 21/50/200 smoothed-MA trend stack held for
# CONFIRM_BARS consecutive 5-min bars. SL distance = 2x the signal candle's
# range; TP distance = SL distance x REWARD_RISK (2.0 = the original 2:1
# rule the live Telegram alert uses; set to 1.0 to test 1:1 instead).
#
# Set REVERSE_SIGNALS = True to test the reversed direction. Note: over a
# full year of real QuantConnect/OANDA data, BOTH the original and reversed
# 2:1 variants lost money (-20.8% and -4.6%) - the earlier +60% "reverse it"
# finding from our own 17-day sample did not hold up and was very likely a
# fluke of that short window, not a real edge. Testing other REWARD_RISK
# values is exploring whether the 2:1 ratio itself was the problem.
#
# Only one position at a time (a new signal is ignored while a trade from a
# previous signal is still open) - this differs slightly from the raw
# backtester (backtester/backtest.js), which opens an independent trade on
# every qualifying bar even if overlapping. Treat this version as closer to
# how the position would actually be managed in a live account.

from AlgorithmImports import *


class CombinedSetupStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(10000)

        self.symbol = self.AddForex("EURUSD", Resolution.Minute, Market.Oanda).Symbol

        # --- config: mirrors telegram-relay/src/strategy.js exactly ---
        self.REVERSE_SIGNALS = False  # flip to True to test the reversed direction
        self.REWARD_RISK = 1.0        # TP distance = SL distance x this. 2.0 = the original 2:1 rule
        self.RISK_PERCENT = 0.01      # fraction of equity risked per trade
        self.MIN_SL_PIPS = 3          # floor so a near-zero candle range can't blow up position size
        self.MAX_LEVERAGE = 10        # safety cap: position notional can't exceed this x equity
        self.RSI_LEN = 14
        self.MA_FAST = 21
        self.MA_MID = 50
        self.MA_SLOW = 200
        self.CONFIRM_BARS = 6
        self.PIP = 0.0001

        self.max_len = self.MA_SLOW + self.CONFIRM_BARS + 20
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []

        # Persisted trend MAs - updated once per bar, never recomputed from
        # a trimmed window (a prior version recomputed SmoothedMA from a
        # ~226-bar rolling buffer every bar, which re-seeds relative to
        # wherever that window currently starts; with only 26 bars of
        # margin over a 200-length MA, ~88% of ma200's value was still the
        # stale re-seed, not the true long-decay average strategy.js
        # computes over full history. These persist across the whole
        # backtest instead.)
        self.ma21_val = None
        self.ma50_val = None
        self.ma200_val = None
        self.ma21_seed = []
        self.ma50_seed = []
        self.ma200_seed = []
        self.trend_hist = []  # last CONFIRM_BARS (close, ma21, ma50, ma200) tuples

        # SL/TP are placed as real StopMarketOrder/LimitOrder brackets (see
        # OnFiveMinuteBar and OnOrderEvent) rather than manually checked
        # against bar ranges - a custom flag trying to guess exact fill
        # timing (two previous attempts) kept reintroducing the same
        # stacking bug via different paths. Whether we're "in a trade" is
        # now read directly from QC's own state every bar: Portfolio
        # quantity and open order count. No separate flag to get wrong.
        self.sl_ticket = None
        self.tp_ticket = None

        consolidator = QuoteBarConsolidator(timedelta(minutes=5))
        consolidator.DataConsolidated += self.OnFiveMinuteBar
        self.SubscriptionManager.AddConsolidator(self.symbol, consolidator)

    # --- indicator math: direct translations of strategy.js ---

    # Incrementally updates one smoothed MA: accumulates a seed buffer until
    # `length` values are in, seeds with their plain average, then applies
    # Wilder-style recursive smoothing forever after - mathematically
    # identical to strategy.js's smoothedMA() run over the full history from
    # bar 0, since both are the same seed-then-recurse rule applied in the
    # same order, just computed incrementally here instead of by batch
    # recomputing an array each time.
    def UpdateSMMA(self, current, seed_buf, new_value, length):
        if current is None:
            seed_buf.append(new_value)
            if len(seed_buf) < length:
                return None, seed_buf
            return sum(seed_buf) / length, []
        return (current * (length - 1) + new_value) / length, seed_buf

    def WilderRSI(self, closes, length):
        n = len(closes)
        rsi = [None] * n
        if n < length + 1:
            return rsi
        gain_sum = 0.0
        loss_sum = 0.0
        for i in range(1, length + 1):
            change = closes[i] - closes[i - 1]
            if change >= 0:
                gain_sum += change
            else:
                loss_sum -= change
        avg_gain = gain_sum / length
        avg_loss = loss_sum / length
        rsi[length] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
        for i in range(length + 1, n):
            change = closes[i] - closes[i - 1]
            gain = change if change > 0 else 0
            loss = -change if change < 0 else 0
            avg_gain = (avg_gain * (length - 1) + gain) / length
            avg_loss = (avg_loss * (length - 1) + loss) / length
            rsi[i] = 100 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
        return rsi

    # Trend from the small rolling history of already-computed (persisted)
    # MA values, rather than recomputing MAs from a trimmed closes array.
    def ComputeTrendFromHistory(self):
        if len(self.trend_hist) < self.CONFIRM_BARS:
            return "none"

        all_up = True
        all_down = True
        for close, ma21, ma50, ma200 in self.trend_hist:
            if ma21 is None or ma50 is None or ma200 is None:
                return "none"
            up = close > ma200 and ma21 > ma50 and ma50 > ma200
            down = close < ma200 and ma21 < ma50 and ma50 < ma200
            if not up:
                all_up = False
            if not down:
                all_down = False
        if all_up:
            return "up"
        if all_down:
            return "down"
        return "none"

    def ComputeArrows(self, opens, closes):
        n = len(closes) - 1
        if n < 3:
            return False, False
        o0, o1, o2, o3 = opens[n], opens[n - 1], opens[n - 2], opens[n - 3]
        c0, c1, c2, c3 = closes[n], closes[n - 1], closes[n - 2], closes[n - 3]

        strike3_bull = c3 < o3 and c2 < o2 and c1 < o1 and c0 > o1
        strike3_bear = c3 > o3 and c2 > o2 and c1 > o1 and c0 < o1

        engulf_bull = o0 <= c1 and o0 < o1 and c0 > o1
        engulf_bear = o0 >= c1 and o0 > o1 and c0 < o1

        return (strike3_bull or engulf_bull), (strike3_bear or engulf_bear)

    # Cancel the sibling SL/TP order once one of them fills, so it doesn't
    # sit resting against a position that no longer exists (which would
    # otherwise open an unwanted new position later if price reached it).
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
        self.opens.append(bar.Open)
        self.highs.append(bar.High)
        self.lows.append(bar.Low)
        self.closes.append(bar.Close)
        if len(self.closes) > self.max_len:
            self.opens.pop(0)
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)

        # Update the persisted trend MAs every bar, unconditionally, so they
        # carry forward across the whole backtest instead of being reseeded
        # from a trimmed window.
        self.ma21_val, self.ma21_seed = self.UpdateSMMA(self.ma21_val, self.ma21_seed, bar.Close, self.MA_FAST)
        self.ma50_val, self.ma50_seed = self.UpdateSMMA(self.ma50_val, self.ma50_seed, bar.Close, self.MA_MID)
        self.ma200_val, self.ma200_seed = self.UpdateSMMA(self.ma200_val, self.ma200_seed, bar.Close, self.MA_SLOW)
        self.trend_hist.append((bar.Close, self.ma21_val, self.ma50_val, self.ma200_val))
        if len(self.trend_hist) > self.CONFIRM_BARS:
            self.trend_hist.pop(0)

        # Single source of truth for "are we already exposed to this
        # symbol": actual position size, or an SL/TP bracket order still
        # resting (which stays open for as long as the position is open).
        # No custom flag, no guessing about fill timing.
        if self.Portfolio[self.symbol].Quantity != 0 or len(self.Transactions.GetOpenOrders(self.symbol)) > 0:
            return

        if len(self.closes) < self.MA_SLOW + self.CONFIRM_BARS:
            return

        trend = self.ComputeTrendFromHistory()
        bull_arrow, bear_arrow = self.ComputeArrows(self.opens, self.closes)
        rsi_series = self.WilderRSI(self.closes, self.RSI_LEN)
        current_rsi = rsi_series[-1]
        if current_rsi is None:
            return

        buy_setup = trend == "up" and bull_arrow and current_rsi > 50
        sell_setup = trend == "down" and bear_arrow and current_rsi < 50

        if self.REVERSE_SIGNALS:
            buy_setup, sell_setup = sell_setup, buy_setup

        current_range = bar.High - bar.Low
        if current_range <= 0:
            return

        price = bar.Close

        # Floor prevents a near-zero candle range from producing an absurd
        # position size (e.g. a 0.3-pip range would otherwise size a
        # "1%-risk" trade at 10+ standard lots on a $10k account).
        sl_distance = max(current_range * 2, self.MIN_SL_PIPS * self.PIP)
        tp_distance = sl_distance * self.REWARD_RISK

        equity = self.Portfolio.TotalPortfolioValue
        risk_amount = equity * self.RISK_PERCENT
        quantity = risk_amount / sl_distance

        # Second safety net: cap notional exposure regardless of how the
        # risk math worked out.
        max_quantity = (equity * self.MAX_LEVERAGE) / price
        quantity = min(quantity, max_quantity)

        self.Debug(
            f"{self.Time} ENTRY {'BUY' if buy_setup else 'SELL'} equity={equity:.2f} "
            f"qty={quantity:.0f} sl_dist={sl_distance:.5f} risk$={risk_amount:.2f}"
        )

        if buy_setup:
            self.MarketOrder(self.symbol, quantity)
            self.sl_ticket = self.StopMarketOrder(self.symbol, -quantity, price - sl_distance)
            self.tp_ticket = self.LimitOrder(self.symbol, -quantity, price + tp_distance)
        elif sell_setup:
            self.MarketOrder(self.symbol, -quantity)
            self.sl_ticket = self.StopMarketOrder(self.symbol, quantity, price + sl_distance)
            self.tp_ticket = self.LimitOrder(self.symbol, quantity, price - tp_distance)
