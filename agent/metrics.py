"""
Observability — Prometheus metrics for the agent.
Falls back to no-ops if prometheus_client is not installed.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server  # type: ignore
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not installed — metrics disabled")


class AgentMetrics:
    def __init__(self, port: int = 8000):
        self._port = port
        if _PROMETHEUS_AVAILABLE:
            self.questions_total = Counter(
                "slack_agent_questions_total",
                "Total questions received",
                ["channel"],
            )
            self.answers_total = Counter(
                "slack_agent_answers_total",
                "Total answers generated",
                ["status"],   # success | error
            )
            self.cache_hits = Counter(
                "slack_agent_cache_hits_total",
                "Number of Redis cache hits",
            )
            self.retrieval_latency = Histogram(
                "slack_agent_retrieval_latency_seconds",
                "Time to retrieve top-K results",
                buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            )
            self.llm_latency = Histogram(
                "slack_agent_llm_latency_seconds",
                "Time for LLM to generate answer",
                buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            )
            self.vector_store_size = Gauge(
                "slack_agent_vector_store_size",
                "Number of documents in vector store",
            )
            self.feedback_total = Counter(
                "slack_agent_feedback_total",
                "User feedback reactions",
                ["polarity"],   # positive | negative
            )
        else:
            # No-op stubs
            self.questions_total = _NoopMetric()
            self.answers_total = _NoopMetric()
            self.cache_hits = _NoopMetric()
            self.retrieval_latency = _NoopMetric()
            self.llm_latency = _NoopMetric()
            self.vector_store_size = _NoopMetric()
            self.feedback_total = _NoopMetric()

    def start_server(self) -> None:
        if _PROMETHEUS_AVAILABLE:
            start_http_server(self._port)
            logger.info("Prometheus metrics served on :%d", self._port)

    @contextmanager
    def time_retrieval(self) -> Generator:
        if _PROMETHEUS_AVAILABLE:
            with self.retrieval_latency.time():
                yield
        else:
            yield

    @contextmanager
    def time_llm(self) -> Generator:
        if _PROMETHEUS_AVAILABLE:
            with self.llm_latency.time():
                yield
        else:
            yield


class _NoopMetric:
    """Stub that silently accepts any attribute access or call."""
    def labels(self, **kwargs): return self
    def inc(self, *a, **kw): pass
    def observe(self, *a, **kw): pass
    def set(self, *a, **kw): pass
    def time(self): return _NoopCtx()


class _NoopCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass
