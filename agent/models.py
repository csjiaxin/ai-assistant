"""
Data models shared across the agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SlackMessage:
    """A single Slack message (may be root or reply)."""
    ts: str                        # Slack timestamp (unique ID)
    channel_id: str
    user_id: str
    text: str
    thread_ts: Optional[str] = None   # set if this is a thread reply
    reaction_count: int = 0
    created_at: Optional[datetime] = None

    @property
    def is_thread_root(self) -> bool:
        return self.thread_ts is None or self.thread_ts == self.ts

    @property
    def slack_url(self) -> str:
        """Generate a deep-link URL to this message."""
        ts_clean = self.ts.replace(".", "")
        return f"https://slack.com/archives/{self.channel_id}/p{ts_clean}"


@dataclass
class QAPair:
    """
    A processed Q&A unit ready for embedding.
    question  = root message text
    answer    = concatenated thread replies
    """
    id: str                     # "{channel_id}_{thread_ts}"
    channel_id: str
    thread_ts: str
    question: str
    answer: str
    questioner_id: str
    respondents: list[str] = field(default_factory=list)
    reaction_score: int = 0     # total 👍 reactions across thread
    created_at: Optional[datetime] = None
    slack_url: str = ""

    @property
    def combined_text(self) -> str:
        """Text used for embedding — question + answer combined."""
        return f"Question: {self.question}\n\nAnswer: {self.answer}"

    def to_metadata(self) -> dict:
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "thread_ts": self.thread_ts,
            "question": self.question[:500],       # truncate for metadata limits
            "answer": self.answer[:2000],
            "questioner_id": self.questioner_id,
            "reaction_score": self.reaction_score,
            "slack_url": self.slack_url,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


@dataclass
class RetrievedContext:
    """A single retrieved Q&A pair with its similarity score."""
    qa: QAPair
    similarity: float

    def format_for_prompt(self, index: int) -> str:
        score_pct = int(self.similarity * 100)
        return (
            f"[Source {index}] (relevance: {score_pct}%, 👍 {self.qa.reaction_score})\n"
            f"Thread: {self.qa.slack_url}\n"
            f"Q: {self.qa.question}\n"
            f"A: {self.qa.answer}"
        )


@dataclass
class AgentResponse:
    """Final answer produced by the agent."""
    answer: str
    sources: list[RetrievedContext]
    question: str
    from_cache: bool = False
    latency_ms: float = 0.0

    def format_slack_message(self, show_sources: bool = True) -> str:
        """Format the answer for posting to Slack (mrkdwn)."""
        blocks = [self.answer]
        if show_sources and self.sources and self.answer:
            source_lines = []
            for i, ctx in enumerate(self.sources, 1):
                pct = int(ctx.similarity * 100)
                source_lines.append(
                    f"• <{ctx.qa.slack_url}|Source {i}> — {pct}% match, 👍 {ctx.qa.reaction_score}"
                )
            blocks.append("\n*📎 Related discussions:*\n" + "\n".join(source_lines))
        if self.from_cache:
            blocks.append("_⚡ (cached response)_")
        return "\n\n".join(blocks)
