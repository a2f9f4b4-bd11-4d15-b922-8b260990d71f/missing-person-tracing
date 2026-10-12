"""Dependency providers, overridden in tests so the API needs no models."""
from __future__ import annotations

import functools
import os

from tracer.config import Config, load_config
from tracer.store import Store


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config(os.environ.get("TRACER_CONFIG", "/app/config.yaml"))


@functools.lru_cache(maxsize=1)
def get_store() -> Store:
    store = Store(os.environ["TRACER_DSN"])
    store.ensure_schema()
    return store


@functools.lru_cache(maxsize=1)
def get_embedder():
    from tracer.reid import PhotoEmbedder  # lazy: loads models
    return PhotoEmbedder(get_config())
