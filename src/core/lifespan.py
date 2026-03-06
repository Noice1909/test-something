from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from neo4j import GraphDatabase

from src.agent.graph import build_graph
from src.config import settings
from src.core.checkpointer import create_checkpointer
from src.core.circuit_breaker import create_circuit_breakers
from src.core.logging import setup_logging
from src.llm.factory import get_llm_from_settings
from src.services.concept_service import ConceptService
from src.services.cypher_generator import CypherGenerator
from src.services.cypher_validator import CypherValidator
from src.services.entity_extractor import EntityExtractor
from src.services.few_shot_service import FewShotService
from src.services.index_service import IndexService
from src.services.neo4j_service import Neo4jService
from src.services.response_generator import ResponseGenerator
from src.services.schema_service import SchemaService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ────────────────────────────────────────
    setup_logging(settings.ENVIRONMENT)
    logger = structlog.get_logger()
    logger.info("startup_begin", environment=settings.ENVIRONMENT)

    # 1. Neo4j driver
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    driver.verify_connectivity()
    logger.info("neo4j_connected", uri=settings.NEO4J_URI, database=settings.NEO4J_DATABASE)

    # 2. Circuit breakers
    breakers = create_circuit_breakers(settings)

    # 2b. LLM instance (one per config, cached via factory)
    llm = get_llm_from_settings(settings)
    logger.info("llm_created", model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)

    # 3. Core services
    neo4j_svc = Neo4jService(driver, settings.NEO4J_DATABASE, breakers["neo4j"])

    index_svc = IndexService(neo4j_svc)
    await index_svc.discover()
    logger.info("indexes_discovered", count=index_svc.count)

    schema_svc = SchemaService(neo4j_svc, settings.SCHEMA_CACHE_TTL)
    await schema_svc.discover()
    logger.info("schema_discovered", labels=len(schema_svc.labels), rels=len(schema_svc.relationship_types))

    concept_svc = ConceptService(neo4j_svc)
    await concept_svc.discover()
    logger.info("concepts_loaded", count=concept_svc.count, available=concept_svc.available)

    few_shot_svc = FewShotService(settings)
    await few_shot_svc.initialize()
    logger.info("few_shot_initialized", count=few_shot_svc.count)

    # 4. Agent services
    entity_extractor = EntityExtractor(neo4j_svc, index_svc, concept_svc, breakers["ollama"], settings, llm)
    cypher_generator = CypherGenerator(schema_svc, breakers["ollama"], settings, llm)
    cypher_validator = CypherValidator(neo4j_svc, schema_svc)
    response_generator = ResponseGenerator(breakers["ollama"], settings, llm)

    # 5. Checkpointer
    checkpointer = await create_checkpointer(settings)
    logger.info("checkpointer_ready", type="memory" if settings.is_local else "redis")

    # 6. Compile LangGraph
    compiled_graph = build_graph(
        concept_svc=concept_svc,
        entity_extractor=entity_extractor,
        schema_svc=schema_svc,
        few_shot_svc=few_shot_svc,
        cypher_generator=cypher_generator,
        cypher_validator=cypher_validator,
        neo4j_svc=neo4j_svc,
        response_generator=response_generator,
        checkpointer=checkpointer,
    )

    # 7. Store on app.state for DI
    app.state.driver = driver
    app.state.breakers = breakers
    app.state.llm = llm
    app.state.neo4j_svc = neo4j_svc
    app.state.index_svc = index_svc
    app.state.schema_svc = schema_svc
    app.state.concept_svc = concept_svc
    app.state.few_shot_svc = few_shot_svc
    app.state.graph = compiled_graph
    app.state.checkpointer = checkpointer
    app.state.settings = settings

    logger.info("startup_complete")

    yield  # ── APPLICATION RUNS ──

    # ── SHUTDOWN ───────────────────────────────────────
    logger.info("shutdown_begin")
    driver.close()
    # Close checkpointer (handles both sync and async)
    if hasattr(checkpointer, "aclose"):
        await checkpointer.aclose()
    elif hasattr(checkpointer, "close"):
        checkpointer.close()
    elif hasattr(checkpointer, "__aexit__"):
        await checkpointer.__aexit__(None, None, None)
    logger.info("shutdown_complete")
