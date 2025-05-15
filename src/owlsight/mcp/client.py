import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from owlsight.agentic.interfaces import (
    MCPClientInterface,
    MCPToolDefinition,
    MCPToolCallResult,
    MCPResource,
)

@dataclass
class MCPServerStdioConfig:
    """Configuration for an MCP server connected via stdio."""
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None

@dataclass
class MCPServerHttpConfig:
    """Configuration for an MCP server connected via HTTP/SSE."""
    base_url: str # e.g., "http://localhost:8080/mcp"

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str  # A unique name to identify this server configuration
    transport_type: str  # 'stdio' or 'http'
    connection_params: Union[MCPServerStdioConfig, MCPServerHttpConfig]

class MCPClientManager(MCPClientInterface):
    """Manages connections and interactions with multiple MCP servers."""

    def __init__(self, server_configs: List[MCPServerConfig]):
        self._server_configs: Dict[str, MCPServerConfig] = { 
            config.name: config for config in server_configs
        }
        self._sessions: Dict[str, ClientSession] = {}
        self._exit_stack = AsyncExitStack()

    async def _get_session(self, server_name: str) -> ClientSession:
        """Gets or creates a ClientSession for the named server."""
        if server_name not in self._sessions:
            if server_name not in self._server_configs:
                raise ValueError(f"Unknown MCP server name: {server_name}")
            
            config = self._server_configs[server_name]
            
            if config.transport_type == 'stdio':
                if not isinstance(config.connection_params, MCPServerStdioConfig):
                    raise ValueError(f"Invalid stdio config for server {server_name}")
                stdio_params = config.connection_params
                server_params = StdioServerParameters(
                    command=stdio_params.command,
                    args=stdio_params.args,
                    env=stdio_params.env,
                )
                read_stream, write_stream = await self._exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
            elif config.transport_type == 'http':
                if not isinstance(config.connection_params, MCPServerHttpConfig):
                    raise ValueError(f"Invalid http config for server {server_name}")
                http_params = config.connection_params
                read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                    streamablehttp_client(http_params.base_url)
                )
            else:
                raise ValueError(f"Unsupported transport type: {config.transport_type}")

            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self._sessions[server_name] = session
        return self._sessions[server_name]

    async def list_remote_tools(self, server_name: str) -> List[MCPToolDefinition]:
        session = await self._get_session(server_name)
        response = await session.list_tools()
        return [
            MCPToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema or {},
            )
            for tool in response.tools
        ]

    async def call_remote_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> MCPToolCallResult:
        session = await self._get_session(server_name)
        mcp_result = await session.call_tool(tool_name, arguments=arguments)
        
        content_list = []
        if mcp_result.content:
            for block in mcp_result.content:
                if isinstance(block, mcp_types.TextContent):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, mcp_types.ImageContent):
                    # Safely access attributes and use model_dump for pydantic models
                    source_data = {}
                    if hasattr(block.source, 'model_dump'):
                        source_data = block.source.model_dump()
                    elif hasattr(block.source, '__dict__'):
                        source_data = vars(block.source)
                    content_list.append({"type": "image", "source": source_data })
                # Add more types as needed (e.g., ErrorContent, EmbeddedResourceContent)
                else:
                    # Generic fallback, may need refinement
                    fallback_data = {}
                    if hasattr(block, 'model_dump'):
                        fallback_data = block.model_dump()
                    elif hasattr(block, '__dict__'):
                         fallback_data = vars(block)
                    content_list.append(fallback_data)

        return MCPToolCallResult(content=content_list)

    async def read_remote_resource(
        self, server_name: str, resource_uri: str
    ) -> MCPResource:
        session = await self._get_session(server_name)
        content_bytes, mime_type = await session.read_resource(resource_uri)
        return MCPResource(content=content_bytes, mime_type=mime_type)

    async def close_all_sessions(self) -> None:
        """Closes all active MCP sessions and cleans up resources."""
        await self._exit_stack.aclose()
        self._sessions.clear()

# Example Usage (for testing purposes, can be removed later):
async def main():
    # This example assumes you have an MCP echo server running locally via stdio
    # You would need to create a simple echo_server.py script for this to work.
    # Example echo_server.py content:
    # from mcp import Server, types
    # import asyncio
    # app = Server("echo-server")
    # @app.tool()
    # async def echo(message: str) -> list[types.ContentBlock]:
    # return [types.TextContent(type="text", text=f"Echo: {message}")]
    # if __name__ == "__main__":
    #     asyncio.run(app.run_stdio())

    configs = [
        MCPServerConfig(
            name="local_echo_server",
            transport_type="stdio",
            connection_params=MCPServerStdioConfig(command="python", args=["path/to/your/echo_server.py"])
        )
    ]
    manager = MCPClientManager(server_configs=configs)
    try:
        tools = await manager.list_remote_tools("local_echo_server")
        print(f"Available tools on local_echo_server: {tools}")
        if tools:
            result = await manager.call_remote_tool("local_echo_server", "echo", {"message": "Hello MCP!"})
            print(f"Tool call result: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await manager.close_all_sessions()

if __name__ == "__main__":
    # To run this example, you'd need an echo_server.py. 
    # Replace "path/to/your/echo_server.py" in the MCPServerConfig above.
    # For instance, if echo_server.py is in the same directory:
    # import os
    # script_path = os.path.join(os.path.dirname(__file__), "echo_server.py") 
    # then use script_path in MCPServerStdioConfig args.
    # asyncio.run(main())
    print("MCPClientManager defined. Run example main() by uncommenting and setting up an echo server.")

