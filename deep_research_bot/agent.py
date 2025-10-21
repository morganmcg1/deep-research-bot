from typing import Any, Callable

import weave
from pydantic import BaseModel, Field
from deep_research_bot.utils import function_tool, perform_tool_calls, console
from deep_research_bot.tools import call_model


class AgentState(BaseModel):
    """Manages the state of the agent."""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    step: int = Field(default=0)
    final_assistant_content: str | None = None # Populated at the end of a run


class SimpleAgent:
    """A simple agent class with tracing, state, and tool processing."""
    def __init__(self, model_name: str, system_message: str, tools: list[Callable]):
        self.model_name = model_name
        self.system_message = system_message
        self.tools = [function_tool(t) for t in tools] # add schemas to the tools
    
    @weave.op(name="SimpleAgent.step") # Trace each step
    def step(self, state: AgentState) -> AgentState:
        step = state.step + 1
        messages = state.messages
        final_assistant_content = None
        try:
            # call model with tools
            response = call_model(
                model_name=self.model_name, 
                messages=messages, 
                tools=[t.tool_schema for t in self.tools])

            # add the response to the messages
            messages.append(response.model_dump())

            # if the LLM requested tool calls, perform them
            if response.tool_calls:
                # perform the tool calls
                tool_outputs = perform_tool_calls(tools=self.tools, tool_calls=response.tool_calls)
                messages.extend(tool_outputs)

            # LLM gave content response
            else:
                final_assistant_content = response.content
        except Exception as e:
            console.print(f"ERROR in Agent Step: {e}")
            # Add an error message to history to indicate failure
            messages.append({"role": "assistant", "content": f"Agent error in step: {str(e)}"})
            final_assistant_content = f"Agent error in step {step}: {str(e)}"
        return AgentState(messages=messages, step=step, final_assistant_content=final_assistant_content)

    @weave.op(name="SimpleAgent.run")
    def run(self, user_prompt: str, max_turns: int = 10) -> AgentState:
        state = AgentState(messages=[
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_prompt}])
        for _ in range(max_turns):
            console.rule(f"Agent Loop Turn {state.step+1}/{max_turns}")
            state = self.step(state)
            if state.final_assistant_content:
                return state
        return state
