"""Quick check that the MCP server works: lists its tools and screens one image.

Usage (from the project folder):  python mcp_server/test_client.py path/to/xray.png
"""
import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main(image):
    params = StdioServerParameters(command=sys.executable, args=["mcp_server/server.py"])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            res = await s.call_tool("screen_xray", {"image_path": image})
            for c in res.content:
                print(getattr(c, "text", c)[:900])
            res = await s.call_tool("search_guidelines", {"query": "chest X-ray triage"})
            for c in res.content:
                print(getattr(c, "text", c)[:300])

asyncio.run(main(sys.argv[1]))
