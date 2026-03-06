from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import StateGraph

from src.agent.nodes import AgentNodes
from src.agent.state import AgentState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from src.services.concept_service import ConceptService
    from src.services.cypher_generator import CypherGenerator
    from src.services.cypher_validator import CypherValidator
    from src.services.entity_extractor import EntityExtractor
    from src.services.few_shot_service import FewShotService
    from src.services.neo4j_service import Neo4jService
    from src.services.response_generator import ResponseGenerator
    from src.services.schema_service import SchemaService


def build_graph(
    *,
    concept_svc: ConceptService,
    entity_extractor: EntityExtractor,
    schema_svc: SchemaService,
    few_shot_svc: FewShotService,
    cypher_generator: CypherGenerator,
    cypher_validator: CypherValidator,
    neo4j_svc: Neo4jService,
    response_generator: ResponseGenerator,
    checkpointer: BaseCheckpointSaver,
) -> StateGraph:
    """Build and compile the LangGraph agent with checkpointer."""

    nodes = AgentNodes(
        concept_svc=concept_svc,
        entity_extractor=entity_extractor,
        schema_svc=schema_svc,
        few_shot_svc=few_shot_svc,
        cypher_generator=cypher_generator,
        cypher_validator=cypher_validator,
        neo4j_svc=neo4j_svc,
        response_generator=response_generator,
    )

    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("match_concepts", nodes.match_concepts)
    graph.add_node("extract_entities", nodes.extract_entities)
    graph.add_node("map_entities", nodes.map_entities)
    graph.add_node("filter_schema", nodes.filter_schema)
    graph.add_node("retrieve_examples", nodes.retrieve_examples)
    graph.add_node("generate_cypher", nodes.generate_cypher)
    graph.add_node("validate_cypher", nodes.validate_cypher)
    graph.add_node("correct_cypher", nodes.correct_cypher)
    graph.add_node("execute_query", nodes.execute_query)
    graph.add_node("generate_response", nodes.generate_response)

    # Linear flow: steps 1-6
    graph.set_entry_point("match_concepts")
    graph.add_edge("match_concepts", "extract_entities")
    graph.add_edge("extract_entities", "map_entities")
    graph.add_edge("map_entities", "filter_schema")
    graph.add_edge("filter_schema", "retrieve_examples")
    graph.add_edge("retrieve_examples", "generate_cypher")
    graph.add_edge("generate_cypher", "validate_cypher")

    # Conditional: validation → pass/fail/fatal
    graph.add_conditional_edges(
        "validate_cypher",
        nodes.validation_router,
        {
            "pass": "execute_query",
            "fail": "correct_cypher",
            "fatal": "generate_response",
        },
    )

    # Correction loops back to validation
    graph.add_edge("correct_cypher", "validate_cypher")

    # After execution, generate response
    graph.add_edge("execute_query", "generate_response")

    # Finish
    graph.set_finish_point("generate_response")

    return graph.compile(checkpointer=checkpointer)
