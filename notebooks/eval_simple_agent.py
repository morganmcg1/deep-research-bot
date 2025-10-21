
import sys
from pathlib import Path
from dataclasses import dataclass

import weave
import simple_parsing as sp


from deep_research_bot.agent import SimpleAgent
from deep_research_bot.tools import exa_search_and_refine
from deep_research_bot.evaluation.eval import run_evaluation
from deep_research_bot.evaluation.eval_config import EvalConfig

@dataclass
class Args:
    model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    wandb_entity: str = "wandb-applied-ai-team"
    wandb_project: str = "london-workshop-2025"
    trials: int = 2
    limit: int = 20
    weave_parallelism: int = 50
    evaluation_name: str = "SimpleAgent"

if __name__ == "__main__":

    args = sp.parse(Args)

    weave.init(f"{args.wandb_entity}/{args.wandb_project}")

    agent = SimpleAgent(
        model_name=args.model_name,
        system_message="You are an agent that has access to an advanced search engine. Please provide the user with the information they are looking for by using the search tool provided. Make sure to keep the sources. Always use tools to obtain reliable results. Return the final answer in markdown format.",
        tools=[exa_search_and_refine]
    )

    # Add project root to Python path
    project_root = Path.cwd().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


    eval_config = EvalConfig(
        evaluation_name=f"{args.evaluation_name}_{args.model_name}",
        trials=args.trials,
        limit=args.limit,
        weave_parallelism=args.weave_parallelism,
        queries=project_root / "data/prompt_data/query.jsonl",
        reference=project_root / "data/test_data/cleaned_data/reference.jsonl",
        criteria=project_root / "data/criteria_data/criteria.jsonl",   
    )

    results = run_evaluation(
        eval_config=eval_config,
        agent_callable=agent.run,
)