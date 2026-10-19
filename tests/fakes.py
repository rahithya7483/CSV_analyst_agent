"""Scripted stand-in for the LLM client, so the agent loop can be tested offline."""
from types import SimpleNamespace


class RateLimitError(Exception):
    status_code = 429


class FakeClient:
    """script items: ("tool", name, args_dict) | ("text", str) | an Exception instance.
    If `loop_tool` is set, that tool call is returned forever once the script runs out."""

    def __init__(self, script, loop_tool=None):
        self.script = list(script)
        self.loop_tool = loop_tool
        self.calls = 0
        self.last_messages = None
        self.chat = SimpleNamespace(completions=self)

    def create(self, messages, **kwargs):
        self.calls += 1
        self.last_messages = messages
        item = self.script.pop(0) if self.script else self.loop_tool
        if isinstance(item, Exception):
            raise item
        if item[0] == "tool":
            _, name, args = item
            import json
            call = SimpleNamespace(id=f"call_{self.calls}",
                                   function=SimpleNamespace(name=name, arguments=json.dumps(args)))
            message = SimpleNamespace(content=None, tool_calls=[call])
        else:
            message = SimpleNamespace(content=item[1], tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
