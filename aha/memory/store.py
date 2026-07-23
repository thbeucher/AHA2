"""Agent-owned episodic and semantic memory.

Implements assumptions: A7.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Episode:
    """One local remembered prediction episode."""

    context: dict[str, Any]
    prediction: dict[str, Any] | None
    outcome: dict[str, Any]
    timestamp: int


@dataclass
class EpisodicMemory:
    """Bounded local episodic memory with simple forgetting."""

    capacity: int = 128
    episodes: deque[Episode] = field(default_factory=deque)

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        while len(self.episodes) > self.capacity:
            self.episodes.popleft()

    def retrieve_recent(self, limit: int = 5) -> list[Episode]:
        return list(self.episodes)[-limit:]

    def __len__(self) -> int:
        return len(self.episodes)


@dataclass
class SemanticMemory:
    """Symbolic compression of repeated episodes."""

    counts: Counter[str] = field(default_factory=Counter)

    def compress_episode(self, episode: Episode) -> None:
        signature = self._signature(episode.context, episode.outcome)
        self.counts[signature] += 1

    def retrieve_common(self, limit: int = 5) -> list[tuple[str, int]]:
        return self.counts.most_common(limit)

    @staticmethod
    def _signature(context: dict[str, Any], outcome: dict[str, Any]) -> str:
        ctx = tuple(sorted((str(k), str(v)) for k, v in context.items()))
        out = tuple(sorted((str(k), str(v)) for k, v in outcome.items()))
        return f"ctx={ctx}|out={out}"
