import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

class EvaluationMode(str, Enum):
    """Evaluation execution modes."""

    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(kw_only=True)
class EvalConfig:
    """
    Minimal DeepResearch RACE evaluator configuration
    """

    target: Optional[Path] = None  # JSONL of model outputs to score (offline mode)
    queries: Path = Path("data/prompt_data/query.jsonl")
    reference: Path = Path("data/test_data/cleaned_data/reference.jsonl")
    criteria: Path = Path("data/criteria_data/criteria.jsonl")
    language: str = "en"  # Language filter: 'all', 'en', or 'zh'
    limit: Optional[int] = None  # Optional cap on number of prompts
    judge_model: str = "gpt-5-nano-2025-08-07"  # LLM judge model name
    temperature: float = 1.0
    reasoning_effort: str = "low"
    output: Path = Path("race_raw_results.jsonl")
    summary: Path = Path("race_summary.json")
    wandb_entity: Optional[str] = "wandb-applied-ai-team"
    wandb_project: Optional[str] = "london-workshop-2025"
    evaluation_name: str = "deep_research_race_eval"
    trials: int = 1
    max_retries: int = 5
    retry_backoff: float = 1.5
    weave_parallelism: Optional[int] = 20
    mode: EvaluationMode = EvaluationMode.OFFLINE
    debug: bool = False

    def __post_init__(self):
        if self.language not in ["all", "en", "zh"]:
            raise ValueError(
                f"Invalid language: {self.language}. Must be 'all', 'en', or 'zh'"
            )
        if not isinstance(self.mode, EvaluationMode):
            self.mode = EvaluationMode(self.mode)
