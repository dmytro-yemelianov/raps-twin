"""Async decision cache with SQLite backend (aiosqlite)."""

import hashlib
import json
import re
import time
from typing import Optional, Any
from dataclasses import dataclass

import aiosqlite

# Avoid importing router types here to prevent circular imports.

 
@dataclass
class CacheEntry:
    """Cache entry metadata."""

    context_hash: str
    prompt_hash: str
    response: str
    hit_count: int
    created_at: float


class DecisionCache:
    """Async cache for LLM decisions with semantic-ish normalization."""

    def __init__(
        self,
        db_path: str = "decision_cache.db",
        max_entries: int = 10000,
    ):
        """
        Initialize decision cache.

        Args:
            db_path: Path to SQLite database file
            max_entries: Maximum number of cache entries
        """
        self.db_path = db_path
        self.max_entries = max_entries
        self._conn: Optional[aiosqlite.Connection] = None

    async def _init_db(self, conn: aiosqlite.Connection):
        """Initialize SQLite database schema on an existing connection."""
        await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    context_hash TEXT,
                    prompt_hash TEXT,
                    response TEXT,
                    hit_count INTEGER DEFAULT 1,
                    created_at REAL,
                    last_accessed REAL,
                    PRIMARY KEY (context_hash, prompt_hash)
                )
            """
        )
        await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_context ON cache(context_hash)
            """
        )
        await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache(last_accessed)
            """
        )
        await conn.commit()

    async def _get_conn(self) -> aiosqlite.Connection:
        """Get or open a shared connection (handles in-memory special case)."""
        if self._conn is None:
            if self.db_path == ":memory:":
                self._conn = await aiosqlite.connect(
                    "file:eddt_decision_cache?mode=memory&cache=shared", uri=True
                )
            else:
                self._conn = await aiosqlite.connect(self.db_path)
            await self._init_db(self._conn)
        return self._conn

    def _hash_context(self, context: Any) -> str:
        """Create hash from context, ignoring volatile fields."""
        stable_context = {
            "agent_role": context.agent_id.split("-")[0] if "-" in context.agent_id else context.agent_id,
            "decision_type": context.decision_type.value,
            "complexity_bucket": self._bucket_complexity(context.complexity_signals),
        }
        return hashlib.sha256(
            json.dumps(stable_context, sort_keys=True).encode()
        ).hexdigest()[:32]  # Truncate to 32 chars for reasonable key size

    def _hash_prompt(self, prompt: str) -> str:
        """Hash prompt, normalizing variable parts."""
        # Remove timestamps, specific IDs, etc.
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", prompt)
        normalized = re.sub(r"\d{2}:\d{2}", "TIME", normalized)
        normalized = re.sub(r"[a-f0-9]{8}-[a-f0-9]{4}", "UUID", normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _bucket_complexity(self, signals: dict) -> str:
        """Bucket complexity into categories for cache grouping."""
        # Simple heuristic bucketing
        if signals.get("queue_length", 0) > 5:
            return "high_load"
        if signals.get("deadline_days", 30) < 3:
            return "urgent"
        return "normal"

    async def get(self, context: Any, prompt: str) -> Optional[str]:
        """
        Try to get cached response.

        Args:
            context: Decision context
            prompt: Prompt text

        Returns:
            Cached response if found, None otherwise
        """
        context_hash = self._hash_context(context)
        prompt_hash = self._hash_prompt(prompt)

        conn = await self._get_conn()
        async with conn.execute(
                """
                SELECT response FROM cache 
                WHERE context_hash = ? AND prompt_hash = ?
            """,
                (context_hash, prompt_hash),
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            await conn.execute(
                """
                UPDATE cache 
                SET hit_count = hit_count + 1, last_accessed = ?
                WHERE context_hash = ? AND prompt_hash = ?
            """,
                (time.time(), context_hash, prompt_hash),
            )
            await conn.commit()
            return row[0]
        return None

    async def store(self, context: Any, prompt: str, response: str):
        """
        Store decision in cache.

        Args:
            context: Decision context
            prompt: Prompt text
            response: LLM response
        """
        context_hash = self._hash_context(context)
        prompt_hash = self._hash_prompt(prompt)
        now = time.time()

        conn = await self._get_conn()
        await conn.execute(
                """
                INSERT OR REPLACE INTO cache 
                (context_hash, prompt_hash, response, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?)
            """,
                (context_hash, prompt_hash, response, now, now),
        )

            # Prune if over limit (remove least recently accessed entries)
        async with conn.execute("SELECT COUNT(*) FROM cache") as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0
        if count > self.max_entries:
            excess = count - self.max_entries
            await conn.execute(
                """
                DELETE FROM cache WHERE rowid IN (
                    SELECT rowid FROM cache 
                    ORDER BY last_accessed ASC
                    LIMIT ?
                )
            """,
                (excess,),
            )

        await conn.commit()

    async def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        conn = await self._get_conn()
        async with conn.execute(
                """
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(hit_count) as total_hits,
                    AVG(hit_count) as avg_hits
                FROM cache
            """
        ) as cursor:
            row = await cursor.fetchone()
        return {
            "total_entries": (row[0] or 0) if row else 0,
            "total_hits": (row[1] or 0) if row else 0,
            "avg_hits_per_entry": (row[2] or 0.0) if row else 0.0,
        }

    async def close(self):
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
