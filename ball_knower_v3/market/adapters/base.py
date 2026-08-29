"""
Market SOURCE ADAPTER boundary (Build A §3, §25).

An adapter converts one raw odds source into canonical `MarketQuote` objects and
declares, honestly, what that source can and cannot support. The declaration is
what stops Ball Knower from silently treating a reconstructed line as executable
history.

Build A builds the boundary and one honest legacy adapter. It does NOT ingest a
paid odds archive: if genuine timestamped executable history requires a new paid
API key / subscription / purchase / irreversible action, an adapter must raise
`SourceAuthorizationRequired` describing exactly what is needed rather than
improvising another source or fabricating history (spec §25).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..quotes import MarketQuote


class SourceAuthorizationRequired(RuntimeError):
    """Raised when a source needs a paid key/subscription/purchase not present.

    The message must state exactly what is required. Never fall back to another
    source or fabricate history (spec §25).
    """


@dataclass(frozen=True)
class SourceCapabilities:
    """Honest declaration of what a market source provides.

    These flags gate downstream trust. A source that does not provide genuine
    timestamped executable history must declare `timestamped_executable_history =
    False`, which forces every quote it yields to be `reference_only`.
    """
    provides_provider_snapshot_time: bool
    provides_bookmaker_update_time: bool
    provides_executable_price: bool
    provides_suspension_state: bool
    timestamped_executable_history: bool
    notes: str = ""


class MarketSourceAdapter:
    """Abstract base for a market source adapter.

    Subclasses implement `capabilities()` and `iter_quotes()`. The base enforces
    one invariant: a source that is not timestamped executable history may only
    yield `reference_only` quotes. This is checked in `iter_quotes_checked()`,
    which callers should prefer.
    """

    source_name: str = "abstract"

    def capabilities(self) -> SourceCapabilities:
        raise NotImplementedError

    def iter_quotes(self) -> Iterator[MarketQuote]:
        raise NotImplementedError

    def iter_quotes_checked(self) -> Iterator[MarketQuote]:
        """Yield quotes, enforcing that no quote CONTRADICTS the declared caps.

        A source may only surface a field it declares it provides. Fabricating a
        snapshot time, a bookmaker-update time, a suspension state, an executable
        price, or executable-history status that the source does not actually
        supply is refused (spec §10, §22 — no silent fabrication).
        """
        caps = self.capabilities()
        name = self.source_name
        for q in self.iter_quotes():
            if not caps.timestamped_executable_history and not q.reference_only:
                raise ValueError(
                    f"adapter {name!r} declares no timestamped executable history but "
                    f"yielded a non-reference_only quote for {q.game_id}; refusing to "
                    f"let a reconstructed line masquerade as executable"
                )
            if not caps.provides_executable_price and q.price_american is not None:
                raise ValueError(
                    f"adapter {name!r} declares no executable price but yielded a price "
                    f"for {q.game_id}; refusing to invent an executable offer"
                )
            if not caps.provides_provider_snapshot_time and q.provider_snapshot_time is not None:
                raise ValueError(
                    f"adapter {name!r} declares no provider snapshot time but yielded one "
                    f"for {q.game_id}; refusing to fabricate a timestamp"
                )
            if not caps.provides_bookmaker_update_time and q.bookmaker_last_update_time is not None:
                raise ValueError(
                    f"adapter {name!r} declares no bookmaker update time but yielded one "
                    f"for {q.game_id}; refusing to fabricate a timestamp"
                )
            if not caps.provides_suspension_state and q.is_suspended is not None:
                raise ValueError(
                    f"adapter {name!r} declares no suspension state but yielded one for "
                    f"{q.game_id}; refusing to fabricate suspension state"
                )
            yield q
