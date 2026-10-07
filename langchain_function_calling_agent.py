"""
----------------------------------------------------------
Kalpesh's LangChain Function Calling Agent Demo
Week 2 Homework: LangChain Function Calling Agent
----------------------------------------------------------
LangChain agent with Groq function calling: calculator + OpenWeather tools.

Required packages:
    pip install "langchain>=1.0,<2.0" "langchain-core>=1.0,<2.0" \
                "langgraph>=1.0,<2.0" "langsmith>=0.3.0" \
                langchain-groq python-dotenv requests
"""

import ast
import operator
import os
import sys

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Swap for any Groq model that supports tool use, e.g. llama-3.1-70b-versatile
GROQ_MODEL = "openai/gpt-oss-120b"


def _require_env(name: str) -> str:
    """Read a required environment variable; exit with a helpful message if absent."""
    value = os.getenv(name)
    if not value:
        sys.exit(
            f"Error: {name} is not set.\n"
            f"  Add it to a .env file in this directory or export it in your shell."
        )
    return value


GROQ_API_KEY = _require_env("GROQ_API_KEY")
OPENWEATHER_API_KEY = _require_env("OPENWEATHER_API_KEY")

# ---------------------------------------------------------------------------
# Calculator tool — safe arithmetic evaluation via AST (no eval on raw input)
# ---------------------------------------------------------------------------

# Only these AST node types are permitted during evaluation.
# ast.Num was removed in Python 3.12; use getattr so the set builds on any version.
_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp, ast.UnaryOp,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
}
if hasattr(ast, "Num"):
    _ALLOWED_NODES.add(ast.Num)  # type: ignore[attr-defined]

_BINARY_OPS: dict = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
}

_UNARY_OPS: dict = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if type(node) not in _ALLOWED_NODES:
        raise ValueError(f"Unsupported operation: {type(node).__name__}")

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return float(node.value)

    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return float(node.n)  # type: ignore[attr-defined]

    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.left), _safe_eval(node.right))

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_fn(_safe_eval(node.operand))

    raise ValueError(f"Unhandled node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result as a string.

    Supports: addition (+), subtraction (-), multiplication (*), true division (/),
    floor division (//), modulo (%), exponentiation (**), and parentheses.
    Does NOT support variables, function calls, or non-numeric literals.
    Call this whenever the user asks you to compute a mathematical expression.

    Args:
        expression: A mathematical expression string, e.g. "(3 + 4) * 2" or "2 ** 10"
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        # Return an integer string when the result is a whole number
        return str(int(result)) if result == int(result) else str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError) as exc:
        return f"Error evaluating expression: {exc}"


# ---------------------------------------------------------------------------
# Weather tool — current conditions via OpenWeather API
# ---------------------------------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Return the current weather conditions for a given city.

    Queries the OpenWeather Current Weather API (v2.5) and returns the
    temperature in both °C and °F, feels-like temperature, humidity, wind
    speed, and a short weather description.

    Returns a clear error string (rather than raising) if the city is not
    found, the API key is invalid, a network problem occurs, or the server
    returns an unexpected response — so the agent can relay the issue.

    Args:
        city: Name of the city, e.g. "London" or "New York". English names
              work best; the API also accepts ISO 3166 country codes appended
              with a comma, e.g. "Springfield,US".
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",   # °C; we convert to °F ourselves for dual display
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
    except requests.RequestException as exc:
        return f"Network error fetching weather for '{city}': {exc}"

    if resp.status_code == 401:
        return "Error: invalid or missing OpenWeather API key (HTTP 401)."
    if resp.status_code == 404:
        return (
            f"Error: city '{city}' not found. "
            "Check the spelling or try a nearby major city."
        )
    if resp.status_code != 200:
        return f"Error: OpenWeather returned HTTP {resp.status_code} for '{city}'."

    data = resp.json()
    temp_c: float   = data["main"]["temp"]
    temp_f: float   = temp_c * 9 / 5 + 32
    feels_c: float  = data["main"]["feels_like"]
    feels_f: float  = feels_c * 9 / 5 + 32
    humidity: int   = data["main"]["humidity"]
    wind_mps: float = data["wind"]["speed"]
    wind_kph: float = wind_mps * 3.6
    description: str = data["weather"][0]["description"].capitalize()
    country: str    = data["sys"]["country"]

    return (
        f"Weather in {city.title()}, {country}: {description}. "
        f"Temperature: {temp_c:.1f}°C / {temp_f:.1f}°F "
        f"(feels like {feels_c:.1f}°C / {feels_f:.1f}°F). "
        f"Humidity: {humidity}%. "
        f"Wind: {wind_mps:.1f} m/s ({wind_kph:.1f} km/h)."
    )


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
)

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to two tools:\n"
    "  - calculator: for evaluating any arithmetic expression\n"
    "  - get_weather: for current weather in any city\n"
    "Call whichever tools are needed to fully answer the user's question. "
    "For questions that combine weather data with arithmetic, call both tools."
)

agent = create_agent(
    model=llm,
    tools=[calculator, get_weather],
    system_prompt=_SYSTEM_PROMPT,
    debug=True,
)


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def _separator(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print("=" * 60)


def run_query(query: str) -> None:
    """Stream the agent and print tool calls + final answer step by step."""
    _separator(f"Query: {query}")
    print("  [streaming agent steps...]")

    final_answer = ""
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 10},
        stream_mode="updates",
    ):
        for node_name, node_output in chunk.items():
            print(f"  [node: {node_name}]")
            for msg in node_output.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"    [Tool call]   {tc['name']}({tc['args']})")
                elif hasattr(msg, "name") and msg.name:
                    print(f"    [Tool result] {msg.name} → {msg.content}")
                elif msg.content:
                    final_answer = msg.content
                    print(f"    [AI message]  {msg.content[:120]}")

    print(f"\nAnswer: {final_answer}")


# ---------------------------------------------------------------------------
# Main — demonstration queries
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke-test the model directly before entering the interactive loop
    print("Smoke-testing Groq API connection...")
    try:
        ping = llm.invoke("Reply with the single word: OK")
        print(f"  Model responded: {ping.content}\n")
    except Exception as exc:
        sys.exit(f"Model smoke test failed: {exc}")

    print("Agent ready. Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        run_query(query)
