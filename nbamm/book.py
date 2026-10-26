"""Limit order book with price-time priority.

Prices are integer cents in [1, 99] for a binary contract that pays 100 cents
if the home team wins and 0 otherwise. Quantity is in contracts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional


@dataclass
class Order:
    oid: int
    owner: str
    side: str          # "buy" or "sell"
    price: int         # limit price in cents
    qty: int


@dataclass
class Fill:
    price: int         # execution price (resting order's price)
    qty: int
    taker: str
    maker: str
    taker_side: str    # side of the aggressive order


class OrderBook:
    def __init__(self) -> None:
        self.bids: Dict[int, Deque[Order]] = {}   # price -> FIFO queue
        self.asks: Dict[int, Deque[Order]] = {}
        self._next_id = 1
        self._index: Dict[int, Order] = {}

    # ---- queries -------------------------------------------------------
    def best_bid(self) -> Optional[int]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[int]:
        return min(self.asks) if self.asks else None

    def depth(self, side: str) -> int:
        book = self.bids if side == "buy" else self.asks
        return sum(o.qty for q in book.values() for o in q)

    # ---- order entry ---------------------------------------------------
    def submit(self, owner: str, side: str, price: int, qty: int) -> List[Fill]:
        """Submit a limit order. It matches against the opposite side first
        (price-time priority); any remainder rests on the book. A market order
        is just a limit order at 99 (buy) or 1 (sell) that is never left resting
        by the caller (see `market`)."""
        assert side in ("buy", "sell") and qty > 0 and 1 <= price <= 99
        fills: List[Fill] = []
        remaining = qty
        opp = self.asks if side == "buy" else self.bids
        while remaining > 0 and opp:
            best = min(opp) if side == "buy" else max(opp)
            if (side == "buy" and best > price) or (side == "sell" and best < price):
                break
            queue = opp[best]
            while remaining > 0 and queue:
                maker = queue[0]
                traded = min(remaining, maker.qty)
                fills.append(Fill(best, traded, owner, maker.owner, side))
                maker.qty -= traded
                remaining -= traded
                if maker.qty == 0:
                    queue.popleft()
                    self._index.pop(maker.oid, None)
            if not queue:
                del opp[best]
        if remaining > 0:
            order = Order(self._next_id, owner, side, price, remaining)
            self._next_id += 1
            (self.bids if side == "buy" else self.asks).setdefault(price, deque()).append(order)
            self._index[order.oid] = order
        return fills

    def market(self, owner: str, side: str, qty: int) -> List[Fill]:
        """Immediate-or-cancel: take whatever is available, never rest."""
        price = 99 if side == "buy" else 1
        fills = self.submit(owner, side, price, qty)
        filled = sum(f.qty for f in fills)
        if filled < qty:                       # drop the unfilled remainder
            self.cancel_owner(owner, only_side=side, only_price=price)
        return fills

    # ---- cancels -------------------------------------------------------
    def cancel_owner(self, owner: str, only_side: Optional[str] = None,
                     only_price: Optional[int] = None) -> None:
        for side, book in (("buy", self.bids), ("sell", self.asks)):
            if only_side and side != only_side:
                continue
            for px in list(book):
                if only_price is not None and px != only_price:
                    continue
                q = deque(o for o in book[px] if o.owner != owner)
                for o in book[px]:
                    if o.owner == owner:
                        self._index.pop(o.oid, None)
                if q:
                    book[px] = q
                else:
                    del book[px]
