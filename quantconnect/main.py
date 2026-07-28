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

        self.longSL = None
        self.longTP = None
        self.shortSL = None
        self.shortTP = None

        consolidator = QuoteBarConsolidator(timedelta(minutes=5))
        consolidator.DataConsolidated += self.OnFiveMinuteBar
        self.SubscriptionManager.AddConsolidator(self.symbol, consolidator)

    # --- indicator math: direct translations of strategy.js ---

    def SmoothedMA(self, values, length):
        n = len(values)
        out = [None] * n
        for i in range(length - 1, n):
            if out[i - 1] is None:
                out[i] = sum(values[i - length + 1:i + 1]) / length
            else:
                out[i] = (out[i - 1] * (length - 1) + values[i]) / length
        return out

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

    def ComputeTrend(self, closes):
        ma21 = self.SmoothedMA(closes, self.MA_FAST)
        ma50 = self.SmoothedMA(closes, self.MA_MID)
        ma200 = self.SmoothedMA(closes, self.MA_SLOW)
        n = len(closes)
        start = n - self.CONFIRM_BARS
        if start < 0:
            return "none"

        all_up = True
        all_down = True
        for i in range(start, n):
            if ma21[i] is None or ma50[i] is None or ma200[i] is None:
                return "none"
            up = closes[i] > ma200[i] and ma21[i] > ma50[i] and ma50[i] > ma200[i]
            down = closes[i] < ma200[i] and ma21[i] < ma50[i] and ma50[i] < ma200[i]
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

        holding = self.Portfolio[self.symbol]

        # Manage an existing position: check this bar's range against the
        # SL/TP locked in when the trade opened.
        if holding.Invested:
            if holding.IsLong:
                if bar.Low <= self.longSL or bar.High >= self.longTP:
                    self.Liquidate(self.symbol)
            elif holding.IsShort:
                if bar.High >= self.shortSL or bar.Low <= self.shortTP:
                    self.Liquidate(self.symbol)
            return  # don't look for new signals while a trade is open

        if len(self.closes) < self.MA_SLOW + self.CONFIRM_BARS:
            return

        trend = self.ComputeTrend(self.closes)
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

        sl_distance = current_range * 2
        tp_distance = sl_distance * self.REWARD_RISK

        equity = self.Portfolio.TotalPortfolioValue
        risk_amount = equity * self.RISK_PERCENT
        quantity = risk_amount / sl_distance

        if buy_setup:
            self.longSL = price - sl_distance
            self.longTP = price + tp_distance
            self.MarketOrder(self.symbol, quantity)
        elif sell_setup:
            self.shortSL = price + sl_distance
            self.shortTP = price - tp_distance
            self.MarketOrder(self.symbol, -quantity)
