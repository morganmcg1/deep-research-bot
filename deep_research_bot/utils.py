# Global Configuration & Setup
import inspect
import json
from enum import Enum
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console as RichConsole
from typing import Any, Callable, get_type_hints
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from pydantic import BaseModel, Field



class Console(RichConsole):
    def md(self, text): 
        return self.print(Markdown(text))

console = Console()


def _generate_tool_schema(func: Callable) -> dict:
    """Given a Python function, generate a tool-compatible JSON schema.
    Handles basic types and Enums. Assumes docstrings are formatted for arg descriptions.
    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    type_hints = get_type_hints(func)

    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func).split("\\n")[0] if inspect.getdoc(func) else "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    docstring = inspect.getdoc(func)
    param_descriptions = {}
    if docstring:
        args_section = False
        current_param = None
        for line in docstring.split('\\n'):
            line_stripped = line.strip()
            if line_stripped.lower().startswith(("args:", "arguments:", "parameters:")):
                args_section = True
                continue
            if args_section:
                if ":" in line_stripped:
                    param_name, desc = line_stripped.split(":", 1)
                    param_descriptions[param_name.strip()] = desc.strip()
                elif line_stripped and not line_stripped.startswith(" "): # Heuristic: end of args section
                     args_section = False

    for name, param in parameters.items():
        is_required = param.default == inspect.Parameter.empty
        param_type = type_hints.get(name, Any)
        json_type = "string"
        param_schema = {}

        # Basic type mapping
        if param_type == str: json_type = "string"
        elif param_type == int: json_type = "integer"
        elif param_type == float: json_type = "number"
        elif param_type == bool: json_type = "boolean"
        elif hasattr(param_type, '__origin__') and param_type.__origin__ is list: # Handle list[type]
             item_type = param_type.__args__[0] if param_type.__args__ else Any
             if item_type == str: param_schema = {"type": "array", "items": {"type": "string"}}
             elif item_type == int: param_schema = {"type": "array", "items": {"type": "integer"}}
             # Add more list item types if needed
             else: param_schema = {"type": "array", "items": {"type": "string"}} # Default list item type
        elif hasattr(param_type, "__members__") and issubclass(param_type, Enum): # Handle Enum
             json_type = "string"
             param_schema["enum"] = [e.value for e in param_type]

        if not param_schema: # If not set by list or Enum
            param_schema["type"] = json_type

        param_schema["description"] = param_descriptions.get(name, "")

        if param.default != inspect.Parameter.empty and param.default is not None:
             param_schema["default"] = param.default # Note: OpenAI schema doesn't officially use default, but useful metadata

        schema["function"]["parameters"]["properties"][name] = param_schema
        if is_required:
            schema["function"]["parameters"]["required"].append(name)
    return schema

def _get_tool(tools: list[Callable], name: str) -> Callable:
    for t in tools:
        if t.__name__ == name:
            return t
    raise KeyError(f"No tool with name {name} found")


def function_tool(func: Callable) -> Callable:
    """Attaches a tool schema to the function and marks it as a tool.
    Call this *after* defining your function: my_func = function_tool(my_func)
    """
    try:
        func.tool_schema = _generate_tool_schema(func)
        func.is_tool = True # Mark it as a tool
    except Exception as e:
        console.print(f"Error processing tool {func.__name__}: {e}")
        # Optionally raise or mark as failed
        func.tool_schema = None
        func.is_tool = False
    return func

def perform_tool_calls(tools: list[Callable], tool_calls: list[ChatCompletionMessageFunctionToolCall]) -> list[dict]:
    "Perform the tool calls and return the messages with the tool call results"
    messages = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        with console.status(f"[bold cyan]Executing {function_name}...[/bold cyan]"):
            tool = _get_tool(tools, function_name)
            tool_response = tool(**function_args)
        
        # Create panel content
        panel_content = f"[bold cyan]🔧 Tool Call:[/bold cyan] {function_name}\n\n"
        panel_content += f"[dim]Args: {tool_call.function.arguments}[/dim]\n\n"
        
        if isinstance(tool_response, list):
            panel_content += f"[green]✓[/green] Found {len(tool_response)} results"
        else:
            panel_content += f"[green]✓[/green] {function_name} executed successfully"
        
        console.print(Panel(panel_content, border_style="cyan"))
        
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "content": str(tool_response),
        })
    return messages


class AgentState(BaseModel):
    """Manages the state of the agent."""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    step: int = Field(default=0)
    final_assistant_content: str | None = None # Populated at the end of a run


