from pathlib import Path
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import weave
import simple_parsing as sp

from deep_research_bot.agent import SimpleAgent, AgentState, AgentStateCompaction
from deep_research_bot import tools
from deep_research_bot.evaluation.eval import run_evaluation
from deep_research_bot.evaluation.eval_config import EvalConfig
from deep_research_bot.utils import console

@dataclass
class Args:
    model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    wandb_entity: str = "wandb-applied-ai-team"
    wandb_project: str = "london-workshop-2025"
    trials: int = 2 # Number of trials to run
    limit: int = 20 # Number of prompts to evaluate
    weave_parallelism: int = 10 # Number of parallel requests to the API
    evaluation_name: str = "SimpleAgent" # Name of the evaluation
    tools: list[str] = field(default_factory=lambda: ["exa_search_and_refine"])  # names from deep_research_bot.tools
    max_turns: int = 10 # Max turns for the agent to run
    use_compaction: bool = False # Enable compaction, defaults to False
    max_tokens: int = 128_000 # Max tokens for the agent to trigger compaction
    deep: bool = False # Enable all features from notebook 02_deep_research_agent.ipynb
    compact_model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507" # Model to use for compaction, defaults to 

if __name__ == "__main__":

    args = sp.parse(Args)

    weave.init(f"{args.wandb_entity}/{args.wandb_project}")

    if args.deep:
        selected_tools = tools.DEEP_RESEARCH_AGENT_TOOLS
    else:
        # Resolve tool names to callables from deep_research_bot.tools
        def resolve_tools(tool_names: list[str]) -> list[Callable]:
            resolved: list[Callable] = []
            for name in tool_names:
                tool = getattr(tools, name, None)
                if tool is None:
                    raise ValueError(f"Tool '{name}' not found in deep_research_bot.tools")
                resolved.append(tool)
            return resolved

        selected_tools = resolve_tools(args.tools)

    state_cls = AgentStateCompaction if (args.use_compaction or args.deep) else AgentState

    if args.deep:
        system_message = tools.DEEP_RESEARCH_AGENT_PROMPT
        console.print("Using compaction for deep research agent")
        args.use_compaction = True
    else:
        system_message = (
            "You are an agent that has access to an advanced search engine. "
            "Please provide the user with the information they are looking for by using the search tool provided. "
            "Make sure to keep the sources. Always use tools to obtain reliable results. Return the final answer in markdown format."
        )

    agent = SimpleAgent(
        model_name=args.model_name,
        system_message="You are an agent that has access to an advanced search engine. Please provide the user with the information they are looking for by using the search tool provided. Make sure to keep the sources. Always use tools to obtain reliable results. Return the final answer in markdown format.",
        tools=selected_tools,
        state_class=state_cls,
    )
    
    project_root = Path.cwd()

    eval_config = EvalConfig(
        evaluation_name=f"{args.evaluation_name}_{args.model_name}",
        trials=args.trials,
        limit=args.limit,
        weave_parallelism=args.weave_parallelism,
        queries=project_root / "data/prompt_data/query.jsonl",
        reference=project_root / "data/test_data/cleaned_data/reference.jsonl",
        criteria=project_root / "data/criteria_data/criteria.jsonl",   
    )

    # Prepare agent callable with optional compaction kwargs
    state_kwargs: dict = {}
    if args.use_compaction:
        state_kwargs["max_tokens"] = args.max_tokens
        if args.compact_model_name:
            state_kwargs["compact_model_name"] = args.compact_model_name

    console.print("Running eval with args: ", args)
    console.print("State class: ", state_cls)
    console.print("State kwargs: ", state_kwargs)
    
    agent_callable = partial(agent.run, max_turns=args.max_turns, **state_kwargs)

    results = run_evaluation(
        eval_config=eval_config,
        agent_callable=agent_callable,
    )