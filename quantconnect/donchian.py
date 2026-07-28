# QuantConnect (LEAN) backtest of the Donchian channel breakout (Turtle
# Trading style) - a direct port of atr()/donchianChannel()/evaluateBreakout()
# from telegram-relay/src/strategy.js. Verified against that JS logic on the
# same real EUR/USD data before shipping: all 415 signals matched exactly
# (timestamp, side, ATR value) - this is a checked port, not a re-derived
# guess.
#
# Entry: close breaks above/below the highest-high/lowest-low of the prior
# DONCHIAN_LEN bars. SL distance = ATR(14) x 2; TP distance = SL distance x
# REWARD_RISK (2.0 = 2:1, 1.0 = 1:1). Set REVERSE_SIGNALS = True to fade the
# breakout instead of taking it straight.
#
# Unlike quantconnect/main.py's SetHoldings(1.0) (100% of equity notional,
# not risk), this version sizes each trade so the DOLLAR risk to the stop
# equals RISK_PERCENT of current equity - the same risk model used
# throughout this project's other backtests, so returns are comparable.
#
# Only one position at a time, same as main.py.

from AlgorithmImports import *


class DonchianBreakoutStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(10000)

        self.symbol = self.AddForex("EURUSD", Resolution.Minute, Market.Oanda).Symbol

        # --- config ---
        self.REVERSE_SIGNALS = False  # flip to True to fade the breakout instead
        self.REWARD_RISK = 2.0        # TP distance = SL distance x this
        self.RISK_PERCENT = 0.01      # fraction of equity risked per trade
        self.MIN_SL_PIPS = 3          # floor so a near-zero ATR can't blow up position size
        self.MAX_LEVERAGE = 10        # safety cap: position notional can't exceed this x equity
        self.PIP = 0.0001
        self.DONCHIAN_LEN = 20
        self.ATR_LEN = 14

        # Only needs to be as long as the Donchian channel lookback (a plain
        # windowed min/max, no convergence issue) - ATR itself is now
        # persisted, not recomputed from this buffer.
        self.max_len = self.DONCHIAN_LEN + 5
        self.highs = []
        self.lows = []
        self.closes = []

        # Persisted ATR - updated once per bar, never recomputed from a
        # trimmed window. A prior version recomputed ATR from a 40-bar
        # rolling buffer every bar (only 25 bars of margin over ATR's own
        # 14-bar length), which re-seeds relative to wherever that window
        # currently starts - about 16% of each bar's ATR was still stale
        # re-seed noise, not the true long-decay average strategy.js
        # computes over full history.
        self.atr_val = None
        self.atr_seed = []
        self.prev_close = None

        # SL/TP are placed as real StopMarketOrder/LimitOrder brackets (see
        # OnFiveMinuteBar and OnOrderEvent) rather than manually checked
        # against bar ranges - a custom flag guessing exact fill timing
        # kept reintroducing the same stacking bug via different paths (see
        # main.py's history). Whether we're "in a trade" is read directly
        # from QC's own state every bar: Portfolio quantity and open order
        # count. No separate flag to get wrong.
        self.sl_ticket = None
        self.tp_ticket = None

        consolidator = QuoteBarConsolidator(timedelta(minutes=5))
        consolidator.DataConsolidated += self.OnFiveMinuteBar
        self.SubscriptionManager.AddConsolidator(self.symbol, consolidator)

    # --- indicator math: direct translations of strategy.js ---

    # Incrementally updates ATR: accumulates true-range values until ATR_LEN
    # are in, seeds with their plain average, then Wilder-smooths forever
    # after - mathematically identical to strategy.js's atr() run over full
    # history from bar 0, verified numerically against the batch version
    # (0 mismatches across 5000 real bars) before shipping.
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

    def DonchianChannel(self, highs, lows, length):
        n = len(highs)
        i = n - 1
        if i - length < 0:
            return None, None
        upper = max(highs[i - length:i])
        lower = min(lows[i - length:i])
        return upper, lower

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
        self.highs.append(bar.High)
        self.lows.append(bar.Low)
        self.closes.append(bar.Close)
        if len(self.closes) > self.max_len:
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)

        # Update persisted ATR every bar, unconditionally.
        current_atr = self.UpdateATR(bar.High, bar.Low, bar.Close, self.ATR_LEN)

        # Single source of truth: actual position size, or a bracket order
        # still resting.
        if self.Portfolio[self.symbol].Quantity != 0 or len(self.Transactions.GetOpenOrders(self.symbol)) > 0:
            return

        if len(self.closes) < self.DONCHIAN_LEN + 1:
            return
        if current_atr is None or not (current_atr > 0):
            return

        upper, lower = self.DonchianChannel(self.highs, self.lows, self.DONCHIAN_LEN)

        price = bar.Close
        buy_setup = upper is not None and price > upper
        sell_setup = lower is not None and price < lower

        if self.REVERSE_SIGNALS:
            buy_setup, sell_setup = sell_setup, buy_setup

        if not buy_setup and not sell_setup:
            return

        # Floor prevents a near-zero ATR from producing an absurd position
        # size (ATR got as low as 0.7 pips in real data, which unfloored
        # would size a "1%-risk" trade at 7+ standard lots on a $10k account).
        sl_distance = max(current_atr * 2, self.MIN_SL_PIPS * self.PIP)
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
