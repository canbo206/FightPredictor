"""Shared database configuration for local Python and Docker Compose."""

import os
import psycopg2


def get_connection():
    url = os.getenv("DATABASE_URL")
    if url:
        return psycopg2.connect(url, connect_timeout=5)
    # Retain the original local defaults; environment variables override each.
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        database=os.getenv("PGDATABASE", "ufc_analytics"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "ufc123"),
        port=os.getenv("PGPORT", "5432"), connect_timeout=5,
    )
