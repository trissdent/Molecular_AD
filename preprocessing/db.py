"""Read-only PostgreSQL access for preprocessing."""

import asyncio
import os

import asyncpg
from dotenv import load_dotenv


def db_url(env_var="POSTGRESQL_URL"):
    """Connection string from .env; fails loudly if it is missing."""
    load_dotenv()
    url = os.environ.get(env_var)
    if not url:
        raise ValueError(f"{env_var} is not set (check .env)")
    return url


async def _fetch(query, args, url):
    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch(query, *args)
    finally:
        await conn.close()
    return [dict(r) for r in rows]


def fetch(query, args=(), url=None):
    """Run a SELECT and return a list of dicts."""
    return asyncio.run(_fetch(query, tuple(args), url or db_url()))