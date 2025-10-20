#!/usr/bin/env python3
"""RACE evaluation script instrumented with Weave Evaluations.

Evaluation data and method taken from Deep Research Bench: https://github.com/Ayanami0730/deep_research_bench

Two primary entrypoints are now available:

1. CLI (`python evaluation/eval.py --target ...`) behaves like the original
   DeepResearch script and runs a Weaver evaluation using pre-generated model
   outputs from disk.
2. The new `run_weave_evaluation` helper lets callers provide a callable (e.g.,
   `agent.run`) that will be invoked inside the weaved `Model.predict`, so
   agent output is generated on-the-fly during evaluation.

The script still loads DeepResearch JSONL assets (queries, references, criteria)
and continues to write JSONL + summary files for local inspection, but the core
scoring loop now flows through `weave.Evaluation` so that every run is traced in
Weights & Biases Weave.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import itertools
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from tenacity import retry, stop_after_attempt, wait_exponential
import weave
from dotenv import load_dotenv
from .judge_prompts import (
    SCORE_PROMPT_EN,
    SCORE_PROMPT_ZH,
    SYSTEM_PROMPT_EN,
    SYSTEM_PROMPT_ZH,
    JudgeOutput,
)
from .eval_config import EvalConfig, EvaluationMode
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr
from simple_parsing import ArgumentParser

warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*Hub is deprecated.*")
warnings.filterwarnings(
    "ignore", message="The 'warn' method is deprecated, use 'warning' instead"
)

load_dotenv()


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models.
# ---------------------------------------------------------------------------


class TaskRecord(BaseModel):
    """Task record"""

    model_config = {"frozen": True}
    id: Any
    prompt: str
    language: str = "en"


class ArticleRecord(BaseModel):
    """Article record"""

    model_config = {"frozen": True}
    id: Any
    prompt: str
    article: str


class EvaluationResult(BaseModel):
    """Evaluation result"""

    id: Any
    prompt: str
    comprehensiveness: float
    insight: float
    instruction_following: float
    readability: float
    overall_score: float
    raw_judge: JudgeOutput


AgentCallableReturn = Union[str, Dict[str, Any]]
AgentCallable = Callable[
    [str], Union[AgentCallableReturn, Awaitable[AgentCallableReturn]]
]


def _ensure_text_response(response: AgentCallableReturn) -> str:
    """
    Normalize agent callable responses into text, the response can be a string, a dictionary, a BaseModel, or a JSON string.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in (
            "article",
            "answer",
            "generated_text",
            "output",
            "response",
            "text",
        ):
            value = response.get(key)
            if isinstance(value, str):
                return value
    if isinstance(response, BaseModel):
        return response.model_dump_json(exclude_none=True)
    return json.dumps(response, ensure_ascii=False)


def _is_async_callable(callable_like: AgentCallable) -> bool:
    """
    Determine if the provided callable is coroutine based (including partials and functors).
    """
    if asyncio.iscoroutinefunction(callable_like):
        return True
    if isinstance(callable_like, functools.partial):
        return _is_async_callable(callable_like.func)  # type: ignore[arg-type]
    call_attr = getattr(callable_like, "__call__", None)
    if call_attr is not None and asyncio.iscoroutinefunction(call_attr):
        return True
    return False


async def _invoke_agent_callable(
    agent_callable: AgentCallable, prompt: str
) -> Optional[str]:
    """
    Call agent callable (sync or async) and return normalized text.

    Blocking callables are executed in a background thread so that Weave can
    schedule multiple evaluation rows concurrently.
    """
    if _is_async_callable(agent_callable):
        result = agent_callable(prompt)
    else:
        result = await asyncio.to_thread(agent_callable, prompt)

    if inspect.isawaitable(result):
        result = await result  # type: ignore[assignment]

    if result is None:
        return None
    return _ensure_text_response(result)


def _config_to_metadata(config: "EvalConfig") -> Dict[str, Any]:
    """
    Convert EvalConfig dataclass into a JSON-friendly metadata dict.
    """
    metadata: Dict[str, Any] = {}
    for config_field in fields(config):
        value = getattr(config, config_field.name)
        if isinstance(value, Path):
            metadata[config_field.name] = str(value)
        elif isinstance(value, dict):
            metadata[config_field.name] = value.copy()
        else:
            metadata[config_field.name] = value
    return metadata


# ---------------------------------------------------------------------------
# I/O helpers (inlined from utils/io_utils.py etc.).
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load JSONL file.
    """
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def format_criteria_list(criteria_data: Dict[str, Any]) -> str:
    """
    Format evaluation criteria list as JSON string, omitting weights.
    """
    criteria_for_prompt: Dict[str, List[Dict[str, str]]] = {}

    for dim, criterions in criteria_data.get("criterions", {}).items():
        if not isinstance(criterions, list):
            logging.warning("Unexpected criteria list type for %s", dim)
            continue
        filtered: List[Dict[str, str]] = []
        for item in criterions:
            if isinstance(item, dict) and "criterion" in item and "explanation" in item:
                filtered.append(
                    {
                        "criterion": item["criterion"],
                        "explanation": item["explanation"],
                    }
                )
        if filtered:
            criteria_for_prompt[dim] = filtered

    return json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)

@weave.op
def calculate_weighted_scores(
    judge_output: JudgeOutput,
    criteria_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Weighted scoring
    """
    results = {
        "target": {"dims": {}, "total": 0.0},
        "reference": {"dims": {}, "total": 0.0},
    }

    dimension_weights: Dict[str, float] = criteria_data.get("dimension_weight", {})
    criterions: Dict[str, List[Dict[str, Any]]] = criteria_data.get("criterions", {})
    criterion_weights: Dict[str, Dict[str, float]] = {
        dim: {c["criterion"]: c["weight"] for c in items if "weight" in c}
        for dim, items in criterions.items()
    }

    total_target = 0.0
    total_reference = 0.0

    for dim_name in [
        "comprehensiveness",
        "insight",
        "instruction_following",
        "readability",
    ]:
        scores_list = getattr(judge_output, dim_name)

        if dim_name not in dimension_weights or dim_name not in criterion_weights:
            logging.warning("Skipping dimension %s due to missing weights", dim_name)
            continue

        dim_weights = criterion_weights[dim_name]
        dim_target_sum = 0.0
        dim_reference_sum = 0.0
        dim_total_weight = 0.0

        for entry in scores_list:
            criterion_text = entry.criterion
            art1_val = entry.article_1_score
            art2_val = entry.article_2_score

            weight = dim_weights.get(criterion_text)

            if weight is None:
                lowered = criterion_text.lower()
                for key, value in dim_weights.items():
                    if (
                        key.lower() == lowered
                        or lowered in key.lower()
                        or key.lower() in lowered
                    ):
                        weight = value
                        break
            if weight is None:
                weight = sum(dim_weights.values()) / max(len(dim_weights), 1)

            dim_target_sum += art1_val * weight
            dim_reference_sum += art2_val * weight
            dim_total_weight += weight

        if dim_total_weight == 0:
            continue

        dim_target_avg = dim_target_sum / dim_total_weight
        dim_reference_avg = dim_reference_sum / dim_total_weight

        results["target"]["dims"][f"{dim_name}_weighted_avg"] = dim_target_avg
        results["reference"]["dims"][f"{dim_name}_weighted_avg"] = dim_reference_avg

        dim_weight = dimension_weights.get(dim_name, 0.0)
        total_target += dim_target_avg * dim_weight
        total_reference += dim_reference_avg * dim_weight

    results["target"]["total"] = total_target
    results["reference"]["total"] = total_reference
    return results


# ---------------------------------------------------------------------------
# LLM interaction utilities.
# ---------------------------------------------------------------------------


def build_judge_prompt(
    language: str,
    task_prompt: str,
    article_1: str,
    article_2: str,
    criteria_list: str,
) -> str:
    """
    Build judge prompt
    """
    system_prompt = SYSTEM_PROMPT_ZH if language == "zh" else SYSTEM_PROMPT_EN
    template = SCORE_PROMPT_ZH if language == "zh" else SCORE_PROMPT_EN
    # The prompt templates include large JSON snippets with braces, so using
    # ``str.format`` would mistakenly treat those as placeholders. Perform
    # narrow replacements instead so only the intended tokens are swapped.
    replacements = {
        "{task_prompt}": task_prompt,
        "{article_1}": article_1,
        "{article_2}": article_2,
        "{criteria_list}": criteria_list,
    }

    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)

    return system_prompt, template

@weave.op
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
)
def call_judge(
    client: OpenAI,
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float = 1.0,
    reasoning_effort: float = "low",
) -> JudgeOutput:
    """
    Call the LLM judge and return parsed Pydantic model.
    """

    llm_kwargs = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "judge_output",
                "schema": JudgeOutput.model_json_schema()
            }
        }
    }

    if "gpt-5" in model:
        llm_kwargs["reasoning_effort"] = reasoning_effort
    else:
        llm_kwargs["temperature"] = temperature
    
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_output",
                "schema": JudgeOutput.model_json_schema()
            }
        },
    )
    raw_json = response.choices[0].message.content
    return JudgeOutput.model_validate_json(raw_json)


# ---------------------------------------------------------------------------
# Weave model and scorers.
# ---------------------------------------------------------------------------


class DeepResearchWeaveModel(weave.Model):
    """
    Weave model that can either call an agent or consume precomputed answers.
    """

    agent_callable: Optional[AgentCallable] = None
    evaluation_mode: EvaluationMode = EvaluationMode.ONLINE

    @weave.op()
    async def predict(
        self,
        prompt: str,
        candidate_article: str = None,
        row_index: int = 0,
        row_id: Any = None,
        trial_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the agent, judge with OpenAI, and compute normalized scores.
        """
        if self.agent_callable is not None:
            result = await _invoke_agent_callable(
                self.agent_callable, prompt
            )
            if isinstance(result, BaseModel):
                candidate_article = result.final_assistant_content
            elif isinstance(result, str):
                candidate_article = result
            else:
                raise ValueError(
                    "Invalid result type from agent callable in DeepResearchWeaveModel.predict."
                )
        elif candidate_article is not None and self.evaluation_mode == EvaluationMode.OFFLINE:
            pass
        else:
            raise ValueError(
                "No agent callable provided for evaluation."
            )

        result = {
            "row_index": row_index,
            "trial_index": trial_index,
            "id": row_id,
            "prompt": prompt,
            "candidate_article": candidate_article,
        }

        return result

class DeepResearchScorer(weave.Scorer):
    """
    Scorer that surfaces normalized DeepResearch metrics.
    """

    name: str = "deep_research_scores"
    judge_model: str = "gpt-5-nano-2025-08-07"
    judge_prompt: str = ""
    temperature: float = 1.0
    reasoning_effort: str = "low",
    api_key: str = ""
    criteria: Dict[str, Any] = {}

    def _call_judge_sync(self, judge_prompt: str, system_prompt: str) -> JudgeOutput:
        client = OpenAI(api_key=self.api_key or None)
        return call_judge(
            client=client,
            system_prompt=system_prompt,
            prompt=judge_prompt,
            model=self.judge_model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

    @weave.op()
    def score(self, output: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        '''
        Score the output using the judge model.
        '''
        # Prefer per-row criteria if present; fall back to scorer-level default
        criteria = kwargs.get('criteria', self.criteria)
        criteria_list_str = format_criteria_list(criteria)

        language = kwargs.get('language', "en")
        reference_article = kwargs.get('reference_article', "")

        system_prompt, judge_prompt = build_judge_prompt(
            language=language,
            task_prompt=output["prompt"],
            article_1=output["candidate_article"],
            article_2=reference_article,
            criteria_list=criteria_list_str,
        )

        judge_output = self._call_judge_sync(judge_prompt=judge_prompt, system_prompt=system_prompt)
        weighted_scores = calculate_weighted_scores(judge_output=judge_output, criteria_data=criteria)

        normalized_scores = compute_normalized_scores(
            weighted_scores if isinstance(weighted_scores, dict) else {}
        )
        return {
            "normalized_scores": normalized_scores,
            "comprehensiveness": float(normalized_scores.get("comprehensiveness", 0.0)),
            "insight": float(normalized_scores.get("insight", 0.0)),
            "instruction_following": float(
                normalized_scores.get("instruction_following", 0.0)
            ),
            "readability": float(normalized_scores.get("readability", 0.0)),
            "overall": float(normalized_scores.get("overall", 0.0)),
        }

    @weave.op()
    def summarize(self, score_rows: List[Dict[str, float]]) -> Dict[str, float]:
        if not score_rows:
            return {}
        totals = {
            "comprehensiveness": 0.0,
            "insight": 0.0,
            "instruction_following": 0.0,
            "readability": 0.0,
            "overall": 0.0,
        }
        for row in score_rows:
            for key in totals:
                totals[key] += float(row.get(key, 0.0))
        count = max(len(score_rows), 1)
        return {key: value / count for key, value in totals.items()}


# ---------------------------------------------------------------------------
# Evaluation flow.
# ---------------------------------------------------------------------------


def normalize_dimension(target: float, reference: float) -> float:
    """
    Normalize dimension
    """
    denom = target + reference
    if denom <= 0:
        return 0.0
    return target / denom

@weave.op
def compute_normalized_scores(weighted_scores: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute normalized scores for target vs reference dimensions.
    """
    target_block = weighted_scores.get("target", {})
    reference_block = weighted_scores.get("reference", {})
    dims = target_block.get("dims", {}) or {}
    dims_ref = reference_block.get("dims", {}) or {}

    normalized = {
        "comprehensiveness": normalize_dimension(
            float(dims.get("comprehensiveness_weighted_avg", 0.0)),
            float(dims_ref.get("comprehensiveness_weighted_avg", 0.0)),
        ),
        "insight": normalize_dimension(
            float(dims.get("insight_weighted_avg", 0.0)),
            float(dims_ref.get("insight_weighted_avg", 0.0)),
        ),
        "instruction_following": normalize_dimension(
            float(dims.get("instruction_following_weighted_avg", 0.0)),
            float(dims_ref.get("instruction_following_weighted_avg", 0.0)),
        ),
        "readability": normalize_dimension(
            float(dims.get("readability_weighted_avg", 0.0)),
            float(dims_ref.get("readability_weighted_avg", 0.0)),
        ),
    }
    normalized["overall"] = normalize_dimension(
        float(target_block.get("total", 0.0)),
        float(reference_block.get("total", 0.0)),
    )
    return normalized


def build_maps(
    items: Iterable[Dict[str, Any]],
    record_cls: type[BaseModel],
    required_fields: Tuple[str, ...],
) -> Dict[str, Any]:
    """
    Build maps
    """
    mapping: Dict[str, Any] = {}
    for row in items:
        if not all(field in row for field in required_fields):
            continue
        mapping[row["prompt"]] = record_cls.model_validate(
            {field: row[field] for field in required_fields}
        )
    return mapping


def load_inputs(
    queries_path: Path,
    reference_path: Path,
    criteria_path: Path,
    *,
    target_path: Optional[Path] = None,
) -> Tuple[
    List[TaskRecord],
    Dict[str, ArticleRecord],
    Dict[str, ArticleRecord],
    Dict[str, Dict[str, Any]],
]:
    """
    Load inputs
    """
    tasks = [
        TaskRecord.model_validate(
            {
                "id": row.get("id"),
                "prompt": row["prompt"],
                "language": row.get("language", "en"),
            }
        )
        for row in load_jsonl(queries_path)
        if "prompt" in row
    ]

    target_items = load_jsonl(target_path) if target_path is not None else []
    candidate_article_map = build_maps(target_items, ArticleRecord, ("id", "prompt", "article"))
    reference_map = build_maps(
        load_jsonl(reference_path), ArticleRecord, ("id", "prompt", "article")
    )
    criteria_map: Dict[str, Dict[str, Any]] = {
        row["prompt"]: row for row in load_jsonl(criteria_path) if "prompt" in row
    }
    assert len(reference_map) > 0, "No reference map provided"
    assert len(criteria_map) > 0, "No criteria map provided"

    return tasks, candidate_article_map, reference_map, criteria_map


def filter_tasks(
    tasks: List[TaskRecord],
    candidate_article_map: Dict[str, ArticleRecord],
    reference_map: Dict[str, ArticleRecord],
    criteria_map: Dict[str, Dict[str, Any]],
    language: Optional[str],
    limit: Optional[int],
    *,
    require_candidate_article: bool,
) -> List[TaskRecord]:
    """
    Filter tasks
    """
    filtered = []
    for task in tasks:
        if language and task.language != language:
            continue
        if require_candidate_article and task.prompt not in candidate_article_map:
            continue
        if task.prompt not in reference_map or task.prompt not in criteria_map:
            continue
        filtered.append(task)
        if limit and len(filtered) >= limit:
            break
    return filtered


def build_evaluation_dataset(
    tasks: List[TaskRecord],
    candidate_article_map: Dict[str, ArticleRecord],
    reference_map: Dict[str, ArticleRecord],
    criteria_map: Dict[str, Dict[str, Any]],
    evaluation_mode: EvaluationMode = EvaluationMode.ONLINE,
    *,
    require_candidate_article: bool,
) -> List[Dict[str, Any]]:
    """
    Construct dataset rows passed into weave.Evaluation.
    """
    dataset_rows: List[Dict[str, Any]] = []
    if len(tasks) == 0:
        raise ValueError("No tasks provided for evaluation")

    for index, task in enumerate(tasks):
        if evaluation_mode == EvaluationMode.OFFLINE:
            candidate_article_entry = candidate_article_map.get(task.prompt)
        else:
            candidate_article_entry = None
        
        reference_entry = reference_map.get(task.prompt)
        criteria_entry = criteria_map.get(task.prompt)
        if reference_entry is None or criteria_entry is None:
            logger.warning(
                "Skipping prompt without reference/criteria: %s", task.prompt
            )
            continue
        if require_candidate_article and candidate_article_entry is None:
            logger.warning(
                "Skipping prompt without precomputed answer in offline mode: %s",
                task.prompt,
            )
            continue
        dataset_rows.append(
            {
                "row_index": index,
                "row_id": task.id if task.id is not None else index,
                "prompt": task.prompt,
                "language": task.language,
                "candidate_article": candidate_article_entry.article if candidate_article_entry else None,
                "reference_article": reference_entry.article,
                "criteria": criteria_entry,
                "trial_index": 0,
            }
        )
    assert len(dataset_rows) > 0, "No dataset rows added during evaluation build"
    return dataset_rows


def outputs_to_evaluation_results(
    outputs: Sequence[Dict[str, Any]],
    dataset_rows: Sequence[Dict[str, Any]],
) -> List[EvaluationResult]:
    """
    Convert weave model outputs into EvaluationResult records, preserving dataset order.
    """
    rows_by_index = {row["row_index"]: row for row in dataset_rows}
    results: List[EvaluationResult] = []
    for output in sorted(
        outputs,
        key=lambda item: (item.get("row_index", 0), item.get("trial_index", 0)),
    ):
        dataset_row = rows_by_index.get(output.get("row_index"))
        if dataset_row is None:
            continue
        normalized = output.get("normalized_scores")
        if not isinstance(normalized, dict):
            normalized = compute_normalized_scores(output.get("weighted_scores", {}))
        judge_output_dict = output.get("judge_output", {})
        results.append(
            EvaluationResult(
                id=dataset_row["row_id"],
                prompt=dataset_row["prompt"],
                comprehensiveness=float(normalized.get("comprehensiveness", 0.0)),
                insight=float(normalized.get("insight", 0.0)),
                instruction_following=float(
                    normalized.get("instruction_following", 0.0)
                ),
                readability=float(normalized.get("readability", 0.0)),
                overall_score=float(normalized.get("overall", 0.0)),
                raw_judge=JudgeOutput.model_validate(judge_output_dict),
            )
        )
    return results


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, float]:
    """
    Aggregate results
    """
    if not results:
        return {}

    def avg(attr: str) -> float:
        return sum(getattr(res, attr) for res in results) / len(results)

    return {
        "comprehensiveness": avg("comprehensiveness"),
        "insight": avg("insight"),
        "instruction_following": avg("instruction_following"),
        "readability": avg("readability"),
        "overall": avg("overall_score"),
    }


def _resolve_wandb_project(config: "EvalConfig") -> Optional[str]:
    if config.wandb_project:
        return config.wandb_project
    if config.wandb_entity and config.wandb_project:
        return f"{config.wandb_entity}/{config.wandb_project}"
    return None


def _prepare_weave_attributes(
    config: "EvalConfig", extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    attributes = {}
    if extra:
        attributes.update(extra)
    attributes.setdefault("evaluation_name", config.evaluation_name)
    attributes.setdefault("judge_model", config.judge_model)
    attributes.setdefault("trials", config.trials)
    attributes.setdefault("evaluation_config", _config_to_metadata(config))
    return attributes


def run_evaluation(
    eval_config: "EvalConfig",
    agent_callable: Optional[AgentCallable] = None,
    weave_attributes: Optional[Dict[str, Any]] = None,
) -> Union[
    Tuple[List[EvaluationResult], Dict[str, float]],
    Awaitable[Tuple[List[EvaluationResult], Dict[str, float]]],
]:
    """
    Run evaluation through weave.Evaluation across online or offline modes.

    When invoked from synchronous code this function returns the evaluation
    results immediately. When invoked from an active asyncio event loop (e.g.,
    inside a notebook cell executed with ``await``) it returns an awaitable
    coroutine so the caller can ``await`` it without triggering
    ``asyncio.run`` nesting errors.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set, please set it in the .env file.")

    if eval_config.debug:
        eval_config.limit = 2
        eval_config.trials = 1
        eval_config.weave_parallelism = 1
        eval_config.evaluation_name = "debug_" + eval_config.evaluation_name
        eval_config.max_retries = 1

    mode_value = eval_config.mode
    mode = (
        mode_value
        if isinstance(mode_value, EvaluationMode)
        else EvaluationMode(mode_value)
    )

    if mode is EvaluationMode.OFFLINE and agent_callable is not None:
        logging.info("agent_callable provided; switching evaluation mode to online")
        mode = EvaluationMode.ONLINE
    if mode is EvaluationMode.ONLINE and agent_callable is None:
        raise ValueError(
            "Online evaluation requires an agent_callable to generate answers."
        )
    if mode is EvaluationMode.OFFLINE and eval_config.target is None:
        raise ValueError(
            "Offline evaluation requires a --target JSONL file of precomputed answers."
        )
    eval_config.mode = mode

    tasks, candidate_article_map, reference_map, criteria_map = load_inputs(
        eval_config.queries,
        eval_config.reference,
        eval_config.criteria,
        target_path=eval_config.target,
    )

    target_language = None if eval_config.language == "all" else eval_config.language
    require_candidate_article = mode is EvaluationMode.OFFLINE
    tasks_to_run = filter_tasks(
        tasks,
        candidate_article_map,
        reference_map,
        criteria_map,
        language=target_language,
        limit=eval_config.limit,
        require_candidate_article=require_candidate_article,
    )

    if not tasks_to_run:
        logging.warning("No tasks matched the provided filters")
        return [], {}

    dataset_rows = build_evaluation_dataset(
        tasks=tasks_to_run,
        candidate_article_map=candidate_article_map,
        reference_map=reference_map,
        criteria_map=criteria_map,
        evaluation_mode=mode,
        require_candidate_article=require_candidate_article,
    )
    if eval_config.debug:
        dataset_rows = dataset_rows[:eval_config.limit] if eval_config.limit else dataset_rows[:2] # pylint: disable=unsubscriptable-object

    
    if not dataset_rows or len(dataset_rows) == 0:
        raise ValueError("No dataset rows constructed for evaluation")
    print(f"{len(dataset_rows)} dataset rows constructed for evaluation")

    if eval_config.weave_parallelism is not None:
        os.environ["WEAVE_PARALLELISM"] = str(eval_config.weave_parallelism)

    init_project = _resolve_wandb_project(eval_config)
    weave.init(init_project)

    attributes = _prepare_weave_attributes(eval_config, weave_attributes)

    model = DeepResearchWeaveModel(
        agent_callable=agent_callable,
    )

    evaluation = weave.Evaluation(
        name=eval_config.evaluation_name,
        dataset=dataset_rows,
        scorers=[DeepResearchScorer(
            judge_model=eval_config.judge_model,
            temperature=eval_config.temperature,
            reasoning_effort=eval_config.reasoning_effort,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )],
        trials=eval_config.trials,
    )

    async def _evaluate() -> Tuple[List[EvaluationResult], Dict[str, float]]:
        with weave.attributes(attributes):
            results = await evaluation.evaluate(
                model,
                __weave={"display_name": eval_config.evaluation_name},  # pylint: disable=not-callable
            )

        # outputs = model._outputs
        # results = outputs_to_evaluation_results(outputs, dataset_rows)
        # results.sort(key=lambda res: res.id if res.id is not None else 1e9)
        # aggregated_summary = aggregate_results(results)
        # return results, aggregated_summary
        return results

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_evaluate())
    return _evaluate()

def save_results(results: List[EvaluationResult], output_path: Path) -> None:
    """
    Save results
    """
    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(item.model_dump_json(exclude_none=True) + "\n")


def save_summary(summary: Dict[str, float], summary_path: Path) -> None:
    """
    Save summary
    """
    if not summary:
        return
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> EvalConfig:
    """
    Parse arguments
    """
    parser = ArgumentParser(description="Minimal DeepResearch RACE evaluator")
    parser.add_arguments(EvalConfig, dest="config")
    eval_config = parser.parse_args(argv)
    return eval_config.config


def configure_logging() -> None:
    """
    Configure logging
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> None:
    """
    Main function
    """
    configure_logging()
    eval_config = parse_args(argv)

    logging.info("Loading inputs and running evaluation...")
    mode_value = (
        eval_config.mode
        if isinstance(eval_config.mode, EvaluationMode)
        else EvaluationMode(eval_config.mode)
    )
    if mode_value is EvaluationMode.ONLINE:
        raise ValueError(
            "CLI execution only supports offline mode. Use evaluate_agent for online runs."
        )

    results, summary = run_evaluation(eval_config)

    if not results:
        logging.warning("No evaluations completed")
        return

    logging.info("Saving raw results to %s", eval_config.output)
    save_results(results, eval_config.output)

    logging.info("Computing summary averages...")
    save_summary(summary, eval_config.summary)

    logging.info("Summary: %s", summary)


def run_weave_evaluation(
    config: EvalConfig,
    *,
    agent_callable: AgentCallable,
    weave_attributes: Optional[Dict[str, Any]] = None,
) -> Union[
    Tuple[List[EvaluationResult], Dict[str, float]],
    Awaitable[Tuple[List[EvaluationResult], Dict[str, float]]],
]:
    """
    Convenience wrapper to kick off the weave evaluation programmatically.
    """
    config.mode = EvaluationMode.ONLINE
    return run_evaluation(
        config, agent_callable=agent_callable, weave_attributes=weave_attributes
    )


def evaluate_agent(
    agent_callable: AgentCallable,
    config: EvalConfig,
    *,
    weave_attributes: Optional[Dict[str, Any]] = None,
) -> Union[
    Tuple[List[EvaluationResult], Dict[str, float]],
    Awaitable[Tuple[List[EvaluationResult], Dict[str, float]]],
]:
    """
    Evaluate a callable (e.g., agent.run) against the configured DeepResearch benchmark.
    """
    config.mode = EvaluationMode.ONLINE
    return run_weave_evaluation(
        config, agent_callable=agent_callable, weave_attributes=weave_attributes
    )


if __name__ == "__main__":
    main()
