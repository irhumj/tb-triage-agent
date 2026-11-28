"""
mcp_server/server.py - exposes the frozen TB screening pipeline as MCP tools.

Any MCP client (our own agent, Claude Desktop, the MCP Inspector) can call:
    screen_xray(image_path)        -> decision, calibrated probability, Grad-CAM, warning
    search_guidelines(query, k)    -> passages from WHO TB screening guidance (after the index is built)

The server never changes a decision: it only calls TBScreener.predict() from src/predict.py.

Run (stdio transport, what desktop clients use):
    python mcp_server/server.py
"""

import os
import sys

ORIG_CWD = os.getcwd()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "rag"))
os.chdir(ROOT)                       # so results/ and rag/ paths work wherever the client starts us

from mcp.server.mcpserver import MCPServer   # MCP Python SDK 2.x (called FastMCP in 1.x)

from predict import TBScreener

ALLOWED = (".png", ".jpg", ".jpeg")
MAX_MB = 25
GRADCAM_DIR = os.path.join(ROOT, "results", "gradcam")

server = MCPServer(
    name="tb-triage",
    instructions=(
        "Tools for a TB chest X-ray triage research prototype. Always call screen_xray first. "
        "Never change or override its 'decision' field. Always pass on its 'validation_warning'. "
        "Use search_guidelines for WHO guidance and cite only the passages it returns."
    ),
)

_screener = None


def get_screener():
    """Load the model once, on the first call, then reuse it."""
    global _screener
    if _screener is None:
        _screener = TBScreener(results_dir=os.path.join(ROOT, "results"))
    return _screener


@server.tool()
def screen_xray(image_path: str) -> dict:
    """Screen one frontal chest X-ray for tuberculosis with the frozen model.

    Returns the decision ('TB likely', 'refer' or 'TB unlikely'), the calibrated TB
    probability, the decision thresholds, the recommended next step, a Grad-CAM heatmap
    path and a validation warning. The decision is final and must not be changed.
    """
    path = image_path if os.path.isabs(image_path) else os.path.join(ORIG_CWD, image_path)
    if not path.lower().endswith(ALLOWED):
        return {"error": f"Only {', '.join(ALLOWED)} files are accepted."}
    if not os.path.exists(path):
        return {"error": f"File not found: {image_path}"}
    if os.path.getsize(path) > MAX_MB * 1024 * 1024:
        return {"error": f"File larger than {MAX_MB} MB."}
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    cam_path = os.path.join(GRADCAM_DIR, os.path.splitext(os.path.basename(path))[0] + "_gradcam.png")
    try:
        return get_screener().predict(path, gradcam_path=cam_path)
    except Exception as e:          # a tool should report errors, not crash the server
        return {"error": f"Could not screen this image: {e}"}


@server.tool()
def search_guidelines(query: str, k: int = 3) -> list[dict]:
    """Search WHO TB screening guidance. Returns up to k passages with section and page.

    Only cite text that appears in the returned passages.
    """
    try:
        from search import search          # rag/search.py, available once the index is built
    except ImportError:
        return [{"error": "Guideline index not built yet. Run: python rag/build_index.py"}]
    return search(query, k=max(1, min(k, 5)))


if __name__ == "__main__":
    server.run()                     # stdio by default
