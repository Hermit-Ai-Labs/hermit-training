"""Tiny deterministic market simulator for grading trading-bot code.

The point: a model's bot code is graded by *running* it against a reproducible market and a fixed
contract, not by eyeballing it. Kept dependency-free (stdlib + the bot's own code) so generated
code runs under `python3 -I` with nothing importable from the cwd.

Contracts a bot must satisfy are defined in eval_bots.py; this file is the shared environment it
imports. Everything is integer-seeded and deterministic.
"""
from __future__ import annotations
import random


class Book:
    """A one-level order book with a mid that follows a seeded random walk.

    Methods a strategy is allowed to use:
      mid()                      -> current mid price
      best_bid(), best_ask()     -> mid -/+ half the exchange spread
      place(side, price, size)   -> returns an order id; 'buy'/'sell'
      cancel(order_id)           -> remove a resting order
      step()                     -> advance one tick; fills any crossed resting orders
      fills                      -> list of (side, price, size) that executed
    """

    def __init__(self, seed=0, start=100.0, vol=0.002, half_spread_bps=5, ticks=500):
        self.rng = random.Random(seed)
        self._mid = float(start)
        self.vol = vol
        self.half = half_spread_bps / 10000.0
        self.ticks = ticks
        self.t = 0
        self.orders = {}         # id -> (side, price, size)
        self._next = 1
        self.fills = []          # (side, price, size)

    def mid(self):
        return self._mid

    def best_bid(self):
        return self._mid * (1 - self.half)

    def best_ask(self):
        return self._mid * (1 + self.half)

    def place(self, side, price, size):
        assert side in ("buy", "sell")
        oid = self._next
        self._next += 1
        self.orders[oid] = (side, float(price), float(size))
        return oid

    def cancel(self, oid):
        self.orders.pop(oid, None)

    def step(self):
        # Random-walk the mid.
        self._mid *= (1 + self.rng.gauss(0, self.vol))
        bid, ask = self.best_bid(), self.best_ask()
        # A resting buy below the market fills when the ask drops to it; sell fills when bid rises.
        done = []
        for oid, (side, price, size) in self.orders.items():
            if side == "buy" and ask <= price:
                self.fills.append(("buy", price, size)); done.append(oid)
            elif side == "sell" and bid >= price:
                self.fills.append(("sell", price, size)); done.append(oid)
        for oid in done:
            self.orders.pop(oid, None)
        self.t += 1
        return self._mid


def price_series(seed=0, n=500, start=100.0, vol=0.01, drift=0.0):
    """A deterministic price path for backtest-style graders."""
    rng = random.Random(seed)
    p = float(start)
    out = [p]
    for _ in range(n - 1):
        p *= (1 + drift + rng.gauss(0, vol))
        out.append(p)
    return out
