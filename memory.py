"""
Conversation memory helpers.

The original app only ever passed {"input": input_text} to the agent, so
follow-up questions had zero context of the prior turns even though the
prompt template declared a MessagesPlaceholder("agent_scratchpad").
This module converts Streamlit's session_state message list into proper
LangChain BaseMessage history and threads it through as chat_history.
"""
from langchain_core.messages import AIMessage, HumanMessage

MAX_HISTORY_TURNS = 6  # keep last N user/assistant turn-pairs for context


def build_chat_history(messages: list[dict]) -> list:
    """Convert Streamlit session_state 'messages' (list of {role, content})
    into a bounded list of LangChain BaseMessages, excluding the very
    latest user turn (which is passed separately as `input`)."""
    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))

    # Keep only the most recent turns to bound token usage.
    max_messages = MAX_HISTORY_TURNS * 2
    return history[-max_messages:]
