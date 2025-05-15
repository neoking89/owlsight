from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MCPToolDefinition:
    """Represents the definition of a tool available on an MCP server."""
    name: str
    description: Optional[str]
    input_schema: Dict[str, Any]  # JSON schema for tool parameters


@dataclass
class MCPToolCallResult:
    """Represents the result of calling a tool on an MCP server."""
    # The content is often a list of content blocks (e.g., text, image).
    # Representing as List[Dict[str, Any]] for generic interface.
    # e.g., [{'type': 'text', 'text': '...'}, {'type': 'image', 'url': '...'}]
    content: List[Dict[str, Any]]


@dataclass
class MCPResource:
    """Represents a resource retrieved from an MCP server."""
    content: bytes
    mime_type: str


class MCPClientInterface(ABC):
    """
    Interface for a Model Context Protocol (MCP) client manager.
    This manager is responsible for connecting to and interacting with
    external MCP servers to consume their tools and resources.
    """

    @abstractmethod
    async def list_remote_tools(self, server_endpoint: str) -> List[MCPToolDefinition]:
        """
        Lists all tools available from the specified MCP server.

        Args:
            server_endpoint: The URI or identifier of the MCP server.

        Returns:
            A list of MCPToolDefinition objects.
        """
        pass

    @abstractmethod
    async def call_remote_tool(
        self, server_endpoint: str, tool_name: str, arguments: Dict[str, Any]
    ) -> MCPToolCallResult:
        """
        Calls a specific tool on the specified MCP server.

        Args:
            server_endpoint: The URI or identifier of the MCP server.
            tool_name: The name of the tool to call.
            arguments: A dictionary of arguments for the tool.

        Returns:
            An MCPToolCallResult object containing the result from the tool.
        """
        pass

    @abstractmethod
    async def read_remote_resource(
        self, server_endpoint: str, resource_uri: str
    ) -> MCPResource:
        """
        Reads a resource from the specified MCP server.

        Args:
            server_endpoint: The URI or identifier of the MCP server.
            resource_uri: The URI of the resource to read on the server.

        Returns:
            An MCPResource object containing the content and MIME type of the resource.
        """
        pass
