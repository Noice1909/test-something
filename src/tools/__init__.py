"""Neo4j graph tools — organised by category.

The ``TOOL_REGISTRY`` dict maps every tool name to its async callable.
Specialists import this registry and invoke tools by name.
"""

from src.tools.schema_discovery import SCHEMA_DISCOVERY_TOOLS
from src.tools.graph_exploration import GRAPH_EXPLORATION_TOOLS
from src.tools.graph_search import GRAPH_SEARCH_TOOLS
from src.tools.aggregation import AGGREGATION_TOOLS
from src.tools.data_inspection import DATA_INSPECTION_TOOLS
from src.tools.query_execution import QUERY_EXECUTION_TOOLS
from src.tools.metadata import METADATA_TOOLS

TOOL_REGISTRY: dict = {
    **SCHEMA_DISCOVERY_TOOLS,
    **GRAPH_EXPLORATION_TOOLS,
    **GRAPH_SEARCH_TOOLS,
    **AGGREGATION_TOOLS,
    **DATA_INSPECTION_TOOLS,
    **QUERY_EXECUTION_TOOLS,
    **METADATA_TOOLS,
}

__all__ = ["TOOL_REGISTRY"]
