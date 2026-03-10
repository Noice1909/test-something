"""Neo4j graph tools — organised by category.

The ``TOOL_REGISTRY`` dict maps every tool name to its async callable.
Specialists import this registry and invoke tools by name.
"""

# ── Original categories (1-7) ────────────────────────────────────────────────
from src.tools.schema_discovery import SCHEMA_DISCOVERY_TOOLS
from src.tools.graph_exploration import GRAPH_EXPLORATION_TOOLS
from src.tools.graph_search import GRAPH_SEARCH_TOOLS
from src.tools.aggregation import AGGREGATION_TOOLS
from src.tools.data_inspection import DATA_INSPECTION_TOOLS
from src.tools.query_execution import QUERY_EXECUTION_TOOLS
from src.tools.metadata import METADATA_TOOLS

# ── APOC categories (8-15) ───────────────────────────────────────────────────
from src.tools.apoc_schema import APOC_SCHEMA_TOOLS
from src.tools.apoc_exploration import APOC_EXPLORATION_TOOLS
from src.tools.apoc_query import APOC_QUERY_TOOLS
from src.tools.apoc_analysis import (
    APOC_ANALYSIS_TOOLS,
    APOC_TEXT_TOOLS,
    APOC_MAP_TOOLS,
    APOC_COLLECTION_TOOLS,
)
from src.tools.apoc_date import APOC_DATE_TOOLS

# ── Extended categories (18-27) ──────────────────────────────────────────────
from src.tools.schema_metadata_extra import (
    SCHEMA_METADATA_EXTRA_TOOLS,
    QUERY_PERFORMANCE_TOOLS,
)
from src.tools.graph_topology import (
    TOPOLOGY_TOOLS,
    RELATIONSHIP_PATTERN_TOOLS,
    PROPERTY_ANALYSIS_TOOLS,
)
from src.tools.graph_sampling_nav import (
    GRAPH_SAMPLING_TOOLS,
    GRAPH_NAVIGATION_TOOLS,
    GRAPH_COMPARISON_TOOLS,
)
from src.tools.gds_monitoring import GDS_TOOLS, MONITORING_TOOLS

TOOL_REGISTRY: dict = {
    # Original (1-7)
    **SCHEMA_DISCOVERY_TOOLS,
    **GRAPH_EXPLORATION_TOOLS,
    **GRAPH_SEARCH_TOOLS,
    **AGGREGATION_TOOLS,
    **DATA_INSPECTION_TOOLS,
    **QUERY_EXECUTION_TOOLS,
    **METADATA_TOOLS,
    # APOC (8-15)
    **APOC_SCHEMA_TOOLS,
    **APOC_EXPLORATION_TOOLS,
    **APOC_QUERY_TOOLS,
    **APOC_ANALYSIS_TOOLS,
    **APOC_TEXT_TOOLS,
    **APOC_MAP_TOOLS,
    **APOC_COLLECTION_TOOLS,
    **APOC_DATE_TOOLS,
    # Extended (18-27)
    **SCHEMA_METADATA_EXTRA_TOOLS,
    **QUERY_PERFORMANCE_TOOLS,
    **TOPOLOGY_TOOLS,
    **RELATIONSHIP_PATTERN_TOOLS,
    **PROPERTY_ANALYSIS_TOOLS,
    **GRAPH_SAMPLING_TOOLS,
    **GRAPH_NAVIGATION_TOOLS,
    **GRAPH_COMPARISON_TOOLS,
    **GDS_TOOLS,
    **MONITORING_TOOLS,
}

__all__ = ["TOOL_REGISTRY"]
