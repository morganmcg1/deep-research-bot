import os
from datetime import datetime
from typing import Any

import openai
import weave
from dotenv import load_dotenv
from exa_py import Exa
from weave.flow.util import async_foreach

from deep_research_bot.utils import function_tool, console

load_dotenv()

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "wandb-applied-ai-team")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "london-workshop-2025")
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "Qwen/Qwen3-235B-A22B-Instruct-2507")

async_oai_client = openai.AsyncOpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key=os.getenv("WANDB_API_KEY"),
    project=f"{WANDB_ENTITY}/{WANDB_PROJECT}")

oai_client = openai.OpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key=os.getenv("WANDB_API_KEY"),
    project=f"{WANDB_ENTITY}/{WANDB_PROJECT}"
)

exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))

@weave.op
async def async_call_model(model_name: str, messages: list[dict[str, Any]], **kwargs) -> str:
    "Call a model with the given messages and kwargs."
    response = await async_oai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        **kwargs
    )

    return response.choices[0].message


@weave.op
def call_model(model_name: str, messages: list[dict[str, Any]], **kwargs) -> str:
    "Call a model with the given messages and kwargs."
    response = oai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        **kwargs
    )
    return response.choices[0].message


@weave.op
@function_tool
def exa_search(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Perform a search query on the web and retrieve the most relevant URLs and web content.
    
    This function uses the Exa search API to find relevant web pages based on the query
    and returns their titles, text content, and URLs.
    
    Args:
        query: The search query. Use detailed, specific queries for better results.
               The quality of results depends on the specificity of the query.
        num_results: The number of search results to retrieve. Defaults to 5.
    
    Returns:
        A list of dictionaries, each containing:
            - title: The title of the web page
            - text: The text content of the web page
            - url: The URL of the web page
    """
    search_results = exa_client.search_and_contents(query=query, type='auto', num_results=num_results)
    
    output = []
    for result in search_results.results:
        output.append(
            {"title": result.title,
            "text": result.text,
            "url": result.url
            }
        )
    return output

    

@weave.op
@function_tool
async def async_exa_search_and_refine(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Perform a search query on the web and retrieve the most relevant URLs and web content.
    
    This function uses the Exa search API to find relevant web pages based on the query
    and returns their titles, text content, and URLs. It then refines the search results
    using the model and returns the refined results.
    
    Args:
        query: The search query. Use detailed, specific queries for better results.
               The quality of results depends on the specificity of the query.
        num_results: The number of search results to retrieve. Defaults to 5.
    
    Returns:
        A list of dictionaries, each containing:
            - title: The title of the web page
            - text: The text content of the web page
            - url: The URL of the web page
    """
    search_results = exa_client.search_and_contents(query=query, type='auto', num_results=num_results)
    
    @weave.op
    async def refine_search_result(result):
        messages = [
            {"role":"system", "content": f"Your task is to extract from the search results only the info that is relevant to answer the query"},
            {"role": "user", "content": f"- query: {query}\n- Search result: {result.text}"}
        ]
        refined_search = await async_call_model(model_name=DEFAULT_MODEL_NAME, messages=messages)
        return refined_search.content

    output = []
    async for _, result, refined_text in async_foreach(search_results.results, refine_search_result, max_concurrent_tasks=5):
        output.append(
            {"title": result.title,
            "text": refined_text,
            "url": result.url
            }
        )
    return output


@weave.op 
@function_tool # <- we can use the decorator to automatically generate the tool schema
def exa_search_and_refine(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Perform a search query on the web and retrieve the most relevant URLs and web content.
    
    This function uses the Exa search API to find relevant web pages based on the query
    and returns their titles, text content, and URLs.
    
    Args:
        query: The search query. Use detailed, specific queries for better results.
               The quality of results depends on the specificity of the query.
        num_results: The number of search results to retrieve. Defaults to 5.
    
    Returns:
        A list of dictionaries, each containing:
            - title: The title of the web page
            - text: The text content of the web page
            - url: The URL of the web page
    """
    search_results = exa_client.search_and_contents(query=query, type='auto', num_results=num_results)
    
    @weave.op
    def refine_search_result(result, query):
        messages = [
            {"role":"system", "content": f"Your task is to extract from the search results only the info that is relevant to answer the query"},
            {"role": "user", "content": f"- query: {query}\n- Search result: {result}"}
        ]
        refined_search = call_model(model_name=DEFAULT_MODEL_NAME, messages=messages)
        return refined_search.content

    output = []
    for item, result in enumerate(search_results.results):
        console.print(f"Refining result {item+1}")
        refined_text = refine_search_result(result.text, query)
        output.append(
            {"title": result.title,
            "text": refined_text,
            "url": result.url
            }
        )
    return output


## Tools for the Deep Research Agent

# new, simple function to get the current date which can help the agent ground the research in the current time
def _get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

DEEP_RESEARCH_AGENT_PROMPT = """
  You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.                                                                                                        │

  <Task>
  Your job is to use tools to gather information about the user's input topic and write a blog post as an answer.
  You can use any of the tools provided to you to find resources that can help answer the research question.
  You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
  Your response should be a thorough answer to the user's question, citing sources and reasoning, providing an overview of the facts or any gaps in the subject.
  </Task>

  <Available Tools>
  You have access to the following tools:
  1. **clarification_tool**: For asking user clarifying questions if needed. If you have clarifying questions start with this.
  2. **planning_tool**: For planning the research.
  2. **exa_search_and_refine**: For conducting web searches to gather information
  2. **think_tool**: For reflection and strategic planning during research

  **CRITICAL: Use think_tool after each search to reflect on results and plan next steps**
  </Available Tools>

  <Instructions>
  Think like a human researcher with limited time. Follow these steps:
  1. **Read the question carefully** - What specific information does the user need?
  2. **Start with broader searches** - Use broad, comprehensive queries first
  3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
  4. **Execute narrower searches as you gather information** - Fill in the gaps
  5. **Stop when you can answer confidently** - Don't keep searching for perfection
  6. **Provide an answer** - At the end, always provide the answer from your research.
  7. **Write a blog post style answer** - Write a blog post style answer that is indepth, well structured,easy to understand and engaging.
  </Instructions>

  **Stop Immediately When**:
  - You can answer the user's question comprehensively
  - You have 3+ relevant examples/sources for the question
  - Your last 2 searches returned similar information
  </Hard Limits>

  <Show Your Thinking>
  After each search tool call, use think_tool to analyze the results:
  - What key information did I find?
  - What's missing?
  - Do I have enough to answer the question comprehensively?
  - Should I search more or provide my answer?
  </Show Your Thinking>
""".format(date=_get_today_str())

@weave.op
@function_tool
def clarification_tool(clarifying_questions: str) -> str:
  """Use this tool to ask clarifying questions to the user.

  ALWAYS USE THIS TOOL AS SOON AS THE USER SUBMITS A REQUEST. THIS SHOULD BE THE FIRST TOOL CALL.
  
  IMPORTANT: If you can see in the messages history that you have already asked a clarifying question, you almost always do not need to ask another one.

  If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.
  If you need to ask a question, follow these guidelines:
  - Be concise while gathering all necessary information.
  - Only ask max 3 questions.
  - Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
  - Use bullet points or numbered lists if appropriate for clarity. Make sure that this uses markdown formatting and will be rendered correctly if the string output is passed to a markdown renderer.
  - Don't ask for unnecessary information, or information that the user has already provided. If you can see that the user has already provided the information, do not ask for it again.

  This tool will return the user clarifications.
  Args: 
    clarifying_questions: Your questions to the user as a single string. Be concise while gathering all necessary information. Only ask max 3 questions. Use bullet points or numbered lists if appropriate for clarity with markdown formatting. Don't ask for unnecessary information, or information that the user has already provided. If there are acronyms, abbreviations, or unknown terms, ask the user to clarify. This tool will return the user clarifications.
  """
  output = input(clarifying_questions)
  return output


@weave.op
@function_tool
def planning_tool(plan: str) -> str:
  """Tool for planning the research.

  If there are no clarifying questions, use this tool as the first step of the research.

  Args:
    plan: A comprehensive research plan as a single string. Include: (1) Short analysis of user request, (2) Sub-queries broken down from the user's request (e.g., for 'what are 3 heaviest pokemons and their weight combined' -> subqueries: 'what are 3 heaviest pokemons', 'pokemon1 weight', 'pokemon2 weight', 'pokemon3 weight'), and (3) Research approach. Format this as structured text within the parameter.
  """

@weave.op
@function_tool
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection as a single string addressing: (1) Analysis of current findings - What concrete information have I gathered? (2) Gap assessment - What crucial information is still missing? (3) Quality evaluation - Do I have sufficient evidence/examples for a good answer? (4) Strategic decision - Should I continue searching or provide my answer? Use after receiving search results, before deciding next steps, when assessing research gaps, or before concluding research.
    """


DEEP_RESEARCH_AGENT_TOOLS = [planning_tool, think_tool, exa_search_and_refine]