"""
agent/agent.py - the TB triage agent: an LLM (Gemini) that uses our MCP tools.

How one request flows:
  1. Start the MCP server (mcp_server/server.py) and ask it which tools it has.
  2. Describe those tools to Gemini, together with rules (the system instruction).
  3. Loop: Gemini either asks for a tool call, or writes its final explanation.
     Every tool call is executed by the MCP server, never by the LLM itself.
  4. CODE, not the LLM, prints the decision block from screen_xray's own output,
     so the LLM cannot change the decision, the probability or the warning.
  5. CODE checks that every WHO page the LLM cites was actually retrieved.

Usage (from the project folder, with GEMINI_API_KEY set):
    python agent/agent.py samples/internal_test_1_tb0608.png
    python agent/agent.py samples/x.png "Patient is HIV positive. What does WHO recommend?"
    python agent/agent.py samples/x.png --no-llm      # tools only, no API key needed
    python agent/agent.py --list-models               # which Gemini models your key can use
"""

import asyncio
import json
import os
import re
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "mcp_server", "server.py")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MAX_STEPS = 6

SYSTEM = """You are the explanation layer of a tuberculosis (TB) chest X-ray triage research prototype.
Rules:
1. Always call screen_xray first, with the image path you are given.
2. The 'decision' returned by screen_xray is final. Never change it, soften it or contradict it.
   Do not interpret the X-ray yourself; you cannot see it.
3. Then call search_guidelines with a short query that fits the decision and the user's question.
4. Write at most 120 words for a health worker: what the decision means and the next step.
   Support guidance claims ONLY with the returned passages, citing them as (WHO Module 2, section, p. N)
   using the section and pdf_page given. If no passage supports a point, say the guidance was not found.
5. Do not give a diagnosis or treatment plan. Do not repeat the validation warning; it is shown separately."""


# ---------- talking to the MCP server ----------

def tool_result_json(result):
    """MCP returns content blocks; our tools return JSON text. Turn it back into Python data."""
    texts = [c.text for c in result.content if getattr(c, "text", None)]
    if not texts:
        return {"error": "empty tool result"}
    parsed = [json.loads(t) if t.strip()[:1] in "[{" else t for t in texts]
    return parsed[0] if len(parsed) == 1 else parsed


def gemini_tools(mcp_tools):
    """Convert the MCP tool list into Gemini function declarations (same names, descriptions, schemas)."""
    from google.genai import types
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(name=t.name, description=t.description or "",
                                  # MCP SDK 2.x calls it input_schema; 1.x called it inputSchema
                                  parameters_json_schema=getattr(t, "input_schema", None)
                                  or getattr(t, "inputSchema", None))
        for t in mcp_tools
    ])]


# ---------- checks done by code, not by the LLM ----------

def decision_block(screen):
    return (
        "=" * 70 + "\n"
        f"DECISION (from the frozen model, not the LLM): {screen['decision'].upper()}\n"
        f"Calibrated TB probability: {screen['tb_probability']:.1%}\n"
        f"Next step: {screen['next_step']}\n"
        f"WARNING: {screen['validation_warning']}\n"
        + "=" * 70
    )


def citation_check(text, retrieved_pages):
    """Every 'p. N' the LLM writes must be a page it actually received from search_guidelines."""
    cited = {int(n) for n in re.findall(r"\bp(?:age|\.)\s*(\d+)", text, flags=re.I)}
    unsupported = sorted(cited - retrieved_pages)
    return {"cited_pages": sorted(cited), "retrieved_pages": sorted(retrieved_pages),
            "unsupported_citations": unsupported, "grounded": not unsupported}


# ---------- the agent loop ----------

async def run(image_path, question, use_llm=True):
    params = StdioServerParameters(command=sys.executable, args=[SERVER])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            screen, retrieved_pages, trace = None, set(), []

            if not use_llm:                                   # tools only: useful without an API key
                screen = tool_result_json(await session.call_tool("screen_xray", {"image_path": image_path}))
                if "error" in screen:
                    return print("Tool error:", screen["error"])
                hits = tool_result_json(await session.call_tool(
                    "search_guidelines", {"query": f"chest radiography screening {screen['decision']}"}))
                print(decision_block(screen))
                for h in hits if isinstance(hits, list) else []:
                    print(f"- {h.get('section')} (p. {h.get('pdf_page')}): {h.get('text', '')[:160]}...")
                return

            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM, tools=gemini_tools(mcp_tools), temperature=0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
            contents = [types.Content(role="user", parts=[types.Part(
                text=f"Image path: {image_path}\nQuestion: {question}")])]

            answer = ""
            for step in range(1, MAX_STEPS + 1):
                resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
                calls = resp.function_calls or []
                if not calls:                                  # no tool request = final answer
                    answer = resp.text or ""
                    break
                contents.append(resp.candidates[0].content)    # keep the model's tool request in history
                replies = []
                for fc in calls:
                    args = dict(fc.args or {})
                    result = tool_result_json(await session.call_tool(fc.name, args))
                    trace.append({"step": step, "tool": fc.name, "args": args})
                    if fc.name == "screen_xray" and isinstance(result, dict) and "decision" in result:
                        screen = result
                    if fc.name == "search_guidelines" and isinstance(result, list):
                        retrieved_pages |= {h["pdf_page"] for h in result if "pdf_page" in h}
                    replies.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
                contents.append(types.Content(role="user", parts=replies))

            if screen is None:                                 # safety net: the LLM never screened
                screen = tool_result_json(await session.call_tool("screen_xray", {"image_path": image_path}))
                answer = "(The language model did not call the screening tool; explanation omitted.)"
            if "error" in screen:
                return print("Tool error:", screen["error"])

            print(decision_block(screen))
            print("\nEXPLANATION (LLM, grounded in retrieved WHO passages):\n" + answer.strip())
            check = citation_check(answer, retrieved_pages)
            print("\nTOOL CALLS:", " -> ".join(f"{t['tool']}" for t in trace) or "none")
            print("CITATION CHECK:", "all citations retrieved" if check["grounded"]
                  else f"UNSUPPORTED pages {check['unsupported_citations']}")
            return {"screen": screen, "answer": answer, "trace": trace, "citations": check}


def list_models():
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--list-models" in args:
        list_models(); sys.exit()
    use_llm = "--no-llm" not in args
    args = [a for a in args if a != "--no-llm"]
    if not args:
        sys.exit(__doc__)
    if use_llm and not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set GEMINI_API_KEY first (see README), or run with --no-llm.")
    image = os.path.abspath(args[0])
    question = args[1] if len(args) > 1 else "Screen this chest X-ray for TB and explain the result."
    asyncio.run(run(image, question, use_llm))
