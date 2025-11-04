import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "deepseek-ai/DeepSeek-R1-0528")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "london-workshop-2025")

class EvaluationMode(str, Enum):
    """Evaluation execution modes."""

    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(kw_only=True)
class EvalConfig:
    """
    Minimal DeepResearch RACE evaluator configuration
    """

    target: Path | None = None  # JSONL of model outputs to score (offline mode)
    queries: Path = Path("data/prompt_data/query.jsonl")
    reference: Path = Path("data/test_data/cleaned_data/reference.jsonl")
    criteria: Path = Path("data/criteria_data/criteria.jsonl")
    language: str = "en"  # Language filter: 'all', 'en', or 'zh'
    limit: int | None = None  # Optional cap on number of prompts
    judge_model: str = JUDGE_MODEL_NAME  # LLM judge model name
    temperature: float = 1.0
    reasoning_effort: str = "low"
    output: Path = Path("race_raw_results.jsonl")
    summary: Path = Path("race_summary.json")
    wandb_entity: str | None = WANDB_ENTITY
    wandb_project: str | None = WANDB_PROJECT
    evaluation_name: str = "deep_research_race_eval"
    trials: int = 1
    max_retries: int = 5
    retry_backoff: float = 1.5
    weave_parallelism: int | None = 20
    mode: EvaluationMode = EvaluationMode.OFFLINE
    debug: bool = False

    def __post_init__(self):
        if self.language not in ["all", "en", "zh"]:
            raise ValueError(
                f"Invalid language: {self.language}. Must be 'all', 'en', or 'zh'"
            )
        if not isinstance(self.mode, EvaluationMode):
            self.mode = EvaluationMode(self.mode)
