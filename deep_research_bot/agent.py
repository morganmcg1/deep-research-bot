import json
from typing import Any, Callable


import weave
from pydantic import BaseModel, Field
from deep_research_bot.utils import function_tool, perform_tool_calls, console, estimate_token_count
from deep_research_bot.tools import call_model


class AgentState(BaseModel):
    """Manages the state of the agent."""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    step: int = Field(default=0)
    final_assistant_content: str | None = None # Populated at the end of a run


from pydantic import PrivateAttr, BaseModel

MODEL_SMALL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

class AgentStateCompaction(AgentState):
    """Enhanced AgentState with context window management and compaction tracking."""
    max_tokens: int = Field(default=5000)
    compaction_count: int = Field(default=0)
    compact_model_name: str = Field(default=MODEL_SMALL)
    _estimated_tokens: int = PrivateAttr(default=0)
    _threshold: float = PrivateAttr(default=0.8)

    def model_post_init(self, __context: Any) -> None:
        self._estimated_tokens = estimate_token_count(self.messages)
        tokens_before = self._estimated_tokens

        console.print(f"Utilization percentage: {self.utilization_percentage()}%")

        if self._estimated_tokens > (self.max_tokens*self._threshold):
            console.print("Compacting conversation...")
            self.messages = self.compact_conversation()

            self._estimated_tokens = estimate_token_count(self.messages)
            
            # Calculate token savings
            tokens_after = estimate_token_count(self.messages)
            tokens_saved = tokens_before - tokens_after
            console.print(f"   ✓ Saved {tokens_saved:,} tokens ({tokens_before:,} → {tokens_after:,})")
            console.print(f"Utilization percentage: {self.utilization_percentage()}%")


    def utilization_percentage(self) -> float: 
        """
        Calculate how much of the context window is being used.

        Returns: 
            float: Percentage from 0-100
        """
        if self.max_tokens == 0:
            return 0.0
        return (self._estimated_tokens / self.max_tokens) * 100
    
    
    @weave.op(name="compact")
    def compact_conversation(self) -> list[dict[str, Any]]:
        """
        Compact the conversation by summarizing older messages.
        """
        messages = self.messages
        
        # Preserve: system message (index 0), first message with instructions (index 1)
        system_msg = messages[0]
        request_msg = messages[1]
        
        # Create a prompt asking for a concise summary
        summary_messages =[ 
        {   "role": "system",
            "content": """You are compacting a deep research agent's conversation history. 
            Summarize this research conversation history concisely. 
            Preserve:
            - Key findings from web searches (with source URLs if mentioned)
            - Important facts, data points, and statistics
            - Research decisions and reasoning
            - Any identified gaps or areas needing more investigation"""},
        {   "role": "user",
            "content": f"""
            Conversation to summarize:
            {json.dumps(messages[1:], indent=2)}
            Provide a structured, concise summary."""
        }
        ]
        
        # Call the model to generate the summary
        summary_response = call_model(
            model_name=self.compact_model_name,
            messages=summary_messages
        )
        
        # Create the compacted message that replaces the old messages
        summary_msg = {
            "role": "assistant",
            "content": f"# RESEARCH CONTEXT: \n\n{summary_response.content}"
        }
        
        # Build new message history: system + summary + recent messages
        new_messages = [system_msg, request_msg, summary_msg] 

        
        # Return a new compacted message history
        return new_messages


class SimpleAgent:
    """A simple agent class with tracing, state, and tool processing."""
    def __init__(self, model_name: str, system_message: str, tools: list[Callable], state_class: BaseModel = AgentState):
        self.model_name = model_name
        self.system_message = system_message
        self.tools = [function_tool(t) for t in tools] # add schemas to the tools
        self.state_class = state_class
    
    @weave.op(name="SimpleAgent.step")
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
        return self.state_class(messages=messages, step=step, final_assistant_content=final_assistant_content)

    @weave.op(name="SimpleAgent.run")
    def run(self, user_prompt: str, max_turns: int = 10, **state_kwargs: Any) -> AgentState:
        state = self.state_class(messages=[
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_prompt}], **state_kwargs)
        for _ in range(max_turns):
            console.rule(f"Agent Loop Turn {state.step+1}/{max_turns}")
            state = self.step(state)
            if state.final_assistant_content:
                return state
        return state
