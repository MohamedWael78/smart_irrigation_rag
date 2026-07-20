"""
Streams agent execution as a sequence of typed events that a synchronous
Streamlit script can consume, without Streamlit needing to know about
asyncio at all.

LangChain's token-level streaming (`on_chat_model_stream`) is only exposed
through the async `astream_events` API. Streamlit's script model is
synchronous, so we run the async generator on a background thread and
hand events back through a plain `queue.Queue`, which a simple `for`
loop in app.py can consume like any other iterator.

Event kinds yielded:
  ("token", str)              -- a piece of the final answer to append
  ("tool_start", dict)        -- a tool call has begun (name + input)
  ("tool_end", dict)          -- a tool call finished (name + output)
  ("error", str)              -- something went wrong
"""
import asyncio
import queue
import threading


def stream_agent_response(agent_executor, input_text: str, chat_history: list):
    q: "queue.Queue[tuple]" = queue.Queue()
    DONE = object()

    def worker():
        async def consume():
            try:
                async for event in agent_executor.astream_events(
                    {"input": input_text, "chat_history": chat_history},
                    version="v2",
                ):
                    kind = event.get("event")

                    if kind == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        content = getattr(chunk, "content", None)
                        if content:
                            q.put(("token", content))

                    elif kind == "on_tool_start":
                        q.put((
                            "tool_start",
                            {
                                "name": event.get("name", "tool"),
                                "input": event["data"].get("input", {}),
                            },
                        ))

                    elif kind == "on_tool_end":
                        q.put((
                            "tool_end",
                            {
                                "name": event.get("name", "tool"),
                                "output": event["data"].get("output"),
                            },
                        ))
            except Exception as e:  # surface to the UI instead of hanging
                q.put(("error", str(e)))
            finally:
                q.put((DONE, None))

        asyncio.run(consume())

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        kind, payload = q.get()
        if kind is DONE:
            break
        yield kind, payload
