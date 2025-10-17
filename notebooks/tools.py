import openai
import os
from exa_py import Exa
from typing import List, Dict, Any
import weave
from weave.flow.util import async_foreach
from dotenv import load_dotenv

from utils import function_tool

load_dotenv()


MODEL_SMALL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

async_oai_client = openai.AsyncOpenAI(
    base_url='https://api.inference.wandb.ai/v1',
    api_key=os.getenv("WANDB_API_KEY"),
    project="milieu/london-workshop-2025")

exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))

@weave.op
async def async_call_model(model_name: str, messages: List[Dict[str, Any]], **kwargs) -> str:
    "Call a model with the given messages and kwargs."
    response = await async_oai_client.chat.completions.create(
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
        refined_search = await async_call_model(model_name=MODEL_SMALL, messages=messages)
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