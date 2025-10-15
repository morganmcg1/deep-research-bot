#!/usr/bin/env python3
"""Minimal RACE evaluation script extracted from deepresearch_bench_race.py.

This script keeps only the pieces needed to:
  * load task prompts, criteria, reference articles, and model outputs;
  * call an LLM judge with the official DeepResearch RACE scoring prompt;
  * parse the judge's JSON response and turn it into normalized scores.

It inlines the key helper utilities so it can live in an otherwise empty
repository. Copy the companion JSONL data files described in the README section
of this script into the same layout (or point the CLI flags at their new
locations) and provide an OpenAI-compatible API key to run evaluations.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from simple_parsing import ArgumentParser


# ---------------------------------------------------------------------------
# Prompt templates (English + Chinese) copied from prompt/score_prompt_*.py.
# ---------------------------------------------------------------------------

SCORE_PROMPT_EN = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to deeply compare two articles on the same task, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate two research articles written for this task. We will assess the articles across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Articles to Evaluate**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**Evaluation Criteria**
Now, you need to evaluate and compare these two articles based on the following **evaluation criteria list**, providing comparative analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate and compare `<article_1>` and `<article_2>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how each article fulfills the requirements of each criterion.
2.  **Comparative Evaluation**: Analyze how the two articles perform on each criterion, referencing the content and criterion explanation.
3.  **Score Separately**: Based on your comparative analysis, score each article on each criterion (0-10 points).

**Scoring Rules**
For each criterion, score both articles on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**. Start with "Standard 1" and proceed sequentially through all criteria:
</Instruction>

<output_format>
{
    "comprehensiveness": [
        {
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        },
        {
            "criterion": [Text content of the second comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        },
        ...
    ],
    "insight": [
        {
            "criterion": [Text content of the first insight evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        },
        ...
    ],
    ...
}
</output_format>

Now, please evaluate the two articles based on the research task and criteria, providing detailed comparative analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
"""

SCORE_PROMPT_ZH = """
<system_role>你是一名严格、细致、客观的调研文章评估专家。你擅长根据具体的评估标准，深入比较两篇针对同一任务的文章，并给出精确的评分和清晰的理由。</system_role>

<user_prompt>
**任务背景**
有一个深度调研任务，你需要评估针对该任务撰写的两篇调研文章。我们会从以下四个维度评估文章：全面性、洞察力、指令遵循能力和可读性。内容如下：
<task>
"{task_prompt}"
</task>

**待评估文章**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**评估标准**
现在，你需要根据以下**评判标准列表**，逐条评估并比较这两篇文章的表现，输出对比分析，然后给出0-10的分数。每个标准都���有其解释，请仔细理解。

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**你的任务**
请严格按照 `<criteria_list>` 中的**每一条标准**，对比评估 `<article_1>` 和 `<article_2>` 在该标准上的具体表现。你需要：
1.  **逐条分析**：针对列表中的每一条标准，分别思考两篇文章是如何满足该标准要求的。
2.  **对比评估**：结合文章内容与标准解释，对比分析两篇文章在每一条标准上的表现。
3.  **分别打分**：基于你的对比分析，为两篇文章在该条标准上的表现分别打分（0-10分）。

**打分规则**
对每一条标准，分别为两篇文章打分，打分范围为 0-10 分（连续的数值）。分数高低应体现文章在该标准上表现的好坏：
*   0-2分：表现很差。几乎完全不符合标准要求。
*   2-4分：表现较差。少量符合标准要求，但有明显不足。
*   4-6分：表现中等。基本符合标准要求，不好不坏。
*   6-8分：表现较好。大部分符合标准要求，有可取之处。
*   8-10分：表现出色/极好。完全或超预期符合标准要求。

**输出格式要求**
请**严格**按照下列`<output_format>`格式输出每一条标准的评估结果，**不要包含任何其他无关内容、引言或总结**。从"标准1"开始，按顺序输出所有标准的评估：
</Instruction>

<output_format>
{
    "comprehensiveness": [
        {
            "criterion": [全面性维度的第一条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        },
        {
            "criterion": [全面性维度的第二条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        },
        ...
    ],
    "insight": [
        {
            "criterion": [洞察力维度的第一条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        },
        ...
    ],
    ...
}
</output_format>

现在，请根据调研任务和标准，对两篇文章进行评估，并按照上述要求给出详细的对比分析和评分，请确保输出格式遵守上述`<output_format>`，而且保证其中的json格式可以解析，注意所有可能导致json解析错误的要转义的符号。
</user_prompt>
"""


# ---------------------------------------------------------------------------
# Simple dataclass containers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskRecord:
    '''
    Data class for task records.
    '''
    id: Any
    prompt: str
    language: str


@dataclass(frozen=True)
class ArticleRecord:
    '''
    Data class for article records.
    '''
    id: Any
    prompt: str
    article: str


@dataclass
class EvaluationResult:
    '''
    Data class for evaluation results.
    '''
    id: Any
    prompt: str
    comprehensiveness: float
    insight: float
    instruction_following: float
    readability: float
    overall_score: float
    raw_judge: Dict[str, Any]


# ---------------------------------------------------------------------------
# I/O helpers (inlined from utils/io_utils.py etc.).
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    '''
    Load JSONL file.
    '''
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def format_criteria_list(criteria_data: Dict[str, Any]) -> str:
    '''
    Format evaluation criteria list as JSON string, omitting weights.
    '''
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


def extract_json_from_markdown(text: str) -> Optional[str]:
    '''
    Extract JSON from plain responses or fenced blocks
    '''
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end > start:
            candidate = text[start:end].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    if "```" in text:
        start = text.find("```") + len("```")
        end = text.find("```", start)
        if end > start:
            candidate = text[start:end].strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    # Fallback: match from first to last brace.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def calculate_weighted_scores(
    llm_output_json: Dict[str, List[Dict[str, Any]]],
    criteria_data: Dict[str, Any],
) -> Dict[str, Any]:
    '''
    Weighted scoring
    '''
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

    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            logging.warning("Dimension %s is not a list in judge output", dim)
            continue

        if dim not in dimension_weights or dim not in criterion_weights:
            logging.warning("Skipping dimension %s due to missing weights", dim)
            continue

        dim_weights = criterion_weights[dim]
        dim_target_sum = 0.0
        dim_reference_sum = 0.0
        dim_total_weight = 0.0

        for entry in scores_list:
            if not isinstance(entry, dict):
                continue
            criterion_text = entry.get("criterion")
            art1 = entry.get("article_1_score", entry.get("target_score"))
            art2 = entry.get("article_2_score")

            if criterion_text is None or art1 is None:
                continue

            weight = dim_weights.get(criterion_text)

            if weight is None:
                lowered = criterion_text.lower()
                for key, value in dim_weights.items():
                    if key.lower() == lowered or lowered in key.lower() or key.lower() in lowered:
                        weight = value
                        break
            if weight is None:
                weight = sum(dim_weights.values()) / max(len(dim_weights), 1)

            try:
                art1_val = float(art1)
                art2_val = float(art2) if art2 is not None else 0.0
            except (TypeError, ValueError):
                continue

            dim_target_sum += art1_val * weight
            dim_reference_sum += art2_val * weight
            dim_total_weight += weight

        if dim_total_weight == 0:
            continue

        dim_target_avg = dim_target_sum / dim_total_weight
        dim_reference_avg = dim_reference_sum / dim_total_weight

        results["target"]["dims"][f"{dim}_weighted_avg"] = dim_target_avg
        results["reference"]["dims"][f"{dim}_weighted_avg"] = dim_reference_avg

        dim_weight = dimension_weights.get(dim, 0.0)
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
    '''
    Build judge prompt
    '''
    template = SCORE_PROMPT_ZH if language == "zh" else SCORE_PROMPT_EN
    return template.format(
        task_prompt=task_prompt,
        article_1=article_1,
        article_2=article_2,
        criteria_list=criteria_list,
    )


def call_judge(
    client: OpenAI,
    prompt: str,
    model: str,
    max_retries: int = 5,
    backoff: float = 1.5,
) -> Dict[str, Any]:
    '''
    Call the LLM judge and return parsed JSON.
    '''

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.choices[0].message.content or ""
            json_blob = extract_json_from_markdown(raw_text)
            if not json_blob:
                raise ValueError("Judge response did not contain JSON")
            return json.loads(json_blob)
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            sleep_for = backoff ** attempt
            logging.warning("Judge call failed (%s); retrying in %.1fs", exc, sleep_for)
            time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Evaluation flow.
# ---------------------------------------------------------------------------


def normalize_dimension(target: float, reference: float) -> float:
    '''
    Normalize dimension
    '''
    denom = target + reference
    if denom <= 0:
        return 0.0
    return target / denom


def evaluate_single_prompt(
    client: OpenAI,
    model_name: str,
    task: TaskRecord,
    target_article: ArticleRecord,
    reference_article: ArticleRecord,
    criteria: Dict[str, Any],
) -> EvaluationResult:
    '''
    Evaluate single prompt
    '''
    criteria_list_str = format_criteria_list(criteria)
    judge_prompt = build_judge_prompt(
        language=task.language,
        task_prompt=task.prompt,
        article_1=target_article.article,
        article_2=reference_article.article,
        criteria_list=criteria_list_str,
    )

    judge_output = call_judge(client, judge_prompt, model=model_name)
    scores = calculate_weighted_scores(judge_output, criteria)

    dims = scores["target"]["dims"]
    dims_ref = scores["reference"]["dims"]

    comprehensiveness = normalize_dimension(
        dims.get("comprehensiveness_weighted_avg", 0.0),
        dims_ref.get("comprehensiveness_weighted_avg", 0.0),
    )
    insight = normalize_dimension(
        dims.get("insight_weighted_avg", 0.0),
        dims_ref.get("insight_weighted_avg", 0.0),
    )
    instruction_following = normalize_dimension(
        dims.get("instruction_following_weighted_avg", 0.0),
        dims_ref.get("instruction_following_weighted_avg", 0.0),
    )
    readability = normalize_dimension(
        dims.get("readability_weighted_avg", 0.0),
        dims_ref.get("readability_weighted_avg", 0.0),
    )

    overall = normalize_dimension(scores["target"]["total"], scores["reference"]["total"])

    return EvaluationResult(
        id=task.id,
        prompt=task.prompt,
        comprehensiveness=comprehensiveness,
        insight=insight,
        instruction_following=instruction_following,
        readability=readability,
        overall_score=overall,
        raw_judge=judge_output,
    )


def build_maps(
    items: Iterable[Dict[str, Any]],
    record_cls,
    required_fields: Tuple[str, ...],
) -> Dict[str, Any]:
    '''
    Build maps
    '''
    mapping: Dict[str, Any] = {}
    for row in items:
        if not all(field in row for field in required_fields):
            continue
        mapping[row["prompt"]] = record_cls(**{field: row[field] for field in required_fields})
    return mapping


def load_inputs(
    queries_path: Path,
    target_path: Path,
    reference_path: Path,
    criteria_path: Path,
) -> Tuple[List[TaskRecord], Dict[str, ArticleRecord], Dict[str, ArticleRecord], Dict[str, Dict[str, Any]]]:
    '''
    Load inputs
    '''
    tasks = [
        TaskRecord(id=row.get("id"), prompt=row["prompt"], language=row.get("language", "en"))
        for row in load_jsonl(queries_path)
        if "prompt" in row
    ]

    target_map = build_maps(load_jsonl(target_path), ArticleRecord, ("id", "prompt", "article"))
    reference_map = build_maps(load_jsonl(reference_path), ArticleRecord, ("id", "prompt", "article"))
    criteria_map: Dict[str, Dict[str, Any]] = {
        row["prompt"]: row for row in load_jsonl(criteria_path) if "prompt" in row
    }

    return tasks, target_map, reference_map, criteria_map


def filter_tasks(
    tasks: List[TaskRecord],
    target_map: Dict[str, ArticleRecord],
    reference_map: Dict[str, ArticleRecord],
    criteria_map: Dict[str, Dict[str, Any]],
    language: Optional[str],
    limit: Optional[int],
) -> List[TaskRecord]:
    '''
    Filter tasks
    '''
    filtered = []
    for task in tasks:
        if language and task.language != language:
            continue
        if task.prompt not in target_map or task.prompt not in reference_map or task.prompt not in criteria_map:
            continue
        filtered.append(task)
        if limit and len(filtered) >= limit:
            break
    return filtered


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, float]:
    '''
    Aggregate results
    '''
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


def run_evaluation(args: "EvalConfig") -> List[EvaluationResult]:
    '''
    Run evaluation
    '''
    tasks, target_map, reference_map, criteria_map = load_inputs(
        args.queries, args.target, args.reference, args.criteria
    )

    target_language = None if args.language == "all" else args.language
    tasks_to_run = filter_tasks(
        tasks,
        target_map,
        reference_map,
        criteria_map,
        language=target_language,
        limit=args.limit,
    )

    if not tasks_to_run:
        logging.warning("No tasks matched the provided filters")
        return []

    client = OpenAI(api_key=args.api_key)

    results: List[EvaluationResult] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_prompt,
                client,
                args.judge_model,
                task,
                target_map[task.prompt],
                reference_map[task.prompt],
                criteria_map[task.prompt],
            ): task
            for task in tasks_to_run
        }

        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logging.error("Task %s failed: %s", task.id, exc)

    results.sort(key=lambda res: res.id if res.id is not None else 1e9)
    return results


def save_results(results: List[EvaluationResult], output_path: Path) -> None:
    '''
    Save results
    '''
    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            line = {
                "id": item.id,
                "prompt": item.prompt,
                "comprehensiveness": item.comprehensiveness,
                "insight": item.insight,
                "instruction_following": item.instruction_following,
                "readability": item.readability,
                "overall_score": item.overall_score,
                "judge_output": item.raw_judge,
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def save_summary(summary: Dict[str, float], summary_path: Path) -> None:
    '''
    Save summary
    '''
    if not summary:
        return
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


@dataclass
class EvalConfig:
    '''
    Minimal DeepResearch RACE evaluator configuration
    '''

    target: Path  # JSONL of model outputs to score
    queries: Path = Path("data/prompt_data/query.jsonl")
    reference: Path = Path("data/test_data/cleaned_data/reference.jsonl")
    criteria: Path = Path("data/criteria_data/criteria.jsonl")
    language: str = "all"  # Language filter: 'all', 'en', or 'zh'
    limit: Optional[int] = None  # Optional cap on number of prompts
    max_workers: int = 4
    judge_model: str = "gpt-4.1-2025-04-14"  # LLM judge model name
    api_key: Optional[str] = None  # OpenAI API key (falls back to OPENAI_API_KEY env var)
    output: Path = Path("race_raw_results.jsonl")
    summary: Path = Path("race_summary.json")

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Provide an OpenAI API key via --api-key or OPENAI_API_KEY")
        if self.language not in ["all", "en", "zh"]:
            raise ValueError(f"Invalid language: {self.language}. Must be 'all', 'en', or 'zh'")


def parse_args(argv: Optional[List[str]] = None) -> EvalConfig:
    '''
    Parse arguments
    '''
    parser = ArgumentParser(description="Minimal DeepResearch RACE evaluator")
    parser.add_arguments(EvalConfig, dest="config")
    args = parser.parse_args(argv)
    return args.config


def configure_logging() -> None:
    '''
    Configure logging
    '''
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main(argv: Optional[List[str]] = None) -> None:
    '''
    Main function
    '''
    configure_logging()
    args = parse_args(argv)

    logging.info("Loading inputs and running evaluation...")
    results = run_evaluation(args)

    if not results:
        logging.warning("No evaluations completed")
        return

    logging.info("Saving raw results to %s", args.output)
    save_results(results, args.output)

    logging.info("Computing summary averages...")
    summary = aggregate_results(results)
    save_summary(summary, args.summary)

    logging.info("Summary: %s", summary)


if __name__ == "__main__":
    main()

