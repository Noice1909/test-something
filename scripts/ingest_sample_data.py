"""
Populate Neo4j with sample org-chart data for testing the NL Agent.

Usage:
    python -m scripts.ingest_sample_data          # Additive (idempotent)
    python -m scripts.ingest_sample_data --clean   # Drop + recreate everything
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root so we can import src.config
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neo4j import GraphDatabase  # noqa: E402


# ── helpers ───────────────────────────────────────────────────

def _get_connection():
    """Reuse .env settings via src.config; fallback to dotenv."""
    try:
        from src.config import settings
        return settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD, settings.NEO4J_DATABASE
    except Exception:
        from dotenv import load_dotenv
        import os
        load_dotenv(ROOT / ".env")
        return (
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", ""),
            os.getenv("NEO4J_DATABASE", "neo4j"),
        )


def _run(session, cypher: str, **params):
    """Execute a write transaction."""
    session.run(cypher, **params).consume()


# ── data definitions ──────────────────────────────────────────

PERSONS = [
    {"name": "Alice Chen", "title": "VP of Engineering", "email": "alice.chen@acme.com", "hire_year": 2018},
    {"name": "Bob Martinez", "title": "Senior Developer", "email": "bob.martinez@acme.com", "hire_year": 2019},
    {"name": "Carol Johnson", "title": "Data Analyst", "email": "carol.johnson@acme.com", "hire_year": 2020},
    {"name": "David Kim", "title": "Project Manager", "email": "david.kim@acme.com", "hire_year": 2017},
    {"name": "Eva Rossi", "title": "DevOps Engineer", "email": "eva.rossi@acme.com", "hire_year": 2021},
    {"name": "Frank O'Brien", "title": "Marketing Director", "email": "frank.obrien@acme.com", "hire_year": 2016},
    {"name": "Grace Tanaka", "title": "UX Designer", "email": "grace.tanaka@acme.com", "hire_year": 2022},
    {"name": "Hector Ruiz", "title": "Junior Developer", "email": "hector.ruiz@acme.com", "hire_year": 2023},
]

DEPARTMENTS = [
    {"name": "Engineering", "budget": 2500000, "floor": 3},
    {"name": "Marketing", "budget": 1200000, "floor": 2},
    {"name": "Data Science", "budget": 1800000, "floor": 3},
    {"name": "Product", "budget": 900000, "floor": 2},
]

PROJECTS = [
    {"name": "Project Phoenix", "status": "active", "start_year": 2023},
    {"name": "Customer Portal", "status": "active", "start_year": 2022},
    {"name": "Data Pipeline", "status": "completed", "start_year": 2021},
    {"name": "Mobile App Redesign", "status": "planning", "start_year": 2024},
]

SKILLS = [
    {"name": "Python", "category": "programming"},
    {"name": "Machine Learning", "category": "data science"},
    {"name": "UI/UX Design", "category": "design"},
]

LOCATIONS = [
    {"name": "San Francisco HQ", "city": "San Francisco", "country": "USA"},
    {"name": "Austin Office", "city": "Austin", "country": "USA"},
    {"name": "London Office", "city": "London", "country": "UK"},
]

# (person_name, department_name)
WORKS_IN = [
    ("Alice Chen", "Engineering"),
    ("Bob Martinez", "Engineering"),
    ("Carol Johnson", "Data Science"),
    ("David Kim", "Product"),
    ("Eva Rossi", "Engineering"),
    ("Frank O'Brien", "Marketing"),
    ("Grace Tanaka", "Product"),
    ("Hector Ruiz", "Engineering"),
]

# (person_name, project_name)
WORKS_ON = [
    ("Alice Chen", "Project Phoenix"),
    ("Alice Chen", "Customer Portal"),
    ("Bob Martinez", "Project Phoenix"),
    ("Carol Johnson", "Data Pipeline"),
    ("David Kim", "Customer Portal"),
    ("David Kim", "Mobile App Redesign"),
    ("Eva Rossi", "Project Phoenix"),
    ("Eva Rossi", "Data Pipeline"),
    ("Grace Tanaka", "Mobile App Redesign"),
    ("Hector Ruiz", "Project Phoenix"),
]

# (person_name, skill_name)
HAS_SKILL = [
    ("Alice Chen", "Python"),
    ("Bob Martinez", "Python"),
    ("Carol Johnson", "Python"),
    ("Carol Johnson", "Machine Learning"),
    ("Eva Rossi", "Python"),
    ("Grace Tanaka", "UI/UX Design"),
    ("Hector Ruiz", "Python"),
]

# (person_name, location_name)
PERSON_LOCATED_IN = [
    ("Alice Chen", "San Francisco HQ"),
    ("Bob Martinez", "San Francisco HQ"),
    ("Carol Johnson", "Austin Office"),
    ("David Kim", "San Francisco HQ"),
    ("Eva Rossi", "London Office"),
    ("Frank O'Brien", "Austin Office"),
    ("Grace Tanaka", "San Francisco HQ"),
    ("Hector Ruiz", "London Office"),
]

# (department_name, location_name)
DEPT_LOCATED_IN = [
    ("Engineering", "San Francisco HQ"),
    ("Marketing", "Austin Office"),
    ("Data Science", "Austin Office"),
    ("Product", "San Francisco HQ"),
]

# (project_name, department_name)
OWNED_BY = [
    ("Project Phoenix", "Engineering"),
    ("Customer Portal", "Product"),
    ("Data Pipeline", "Data Science"),
    ("Mobile App Redesign", "Product"),
]

# (person_name, department_name) — managers
MANAGES = [
    ("Alice Chen", "Engineering"),
    ("Frank O'Brien", "Marketing"),
    ("David Kim", "Product"),
]

# (person_name, manager_name)
REPORTS_TO = [
    ("Bob Martinez", "Alice Chen"),
    ("Eva Rossi", "Alice Chen"),
    ("Hector Ruiz", "Alice Chen"),
    ("Carol Johnson", "David Kim"),
    ("Grace Tanaka", "David Kim"),
]

CONCEPTS = [
    {
        "name": "Person",
        "nlp_terms": "person,people,employee,employees,staff,worker,workers,team member,developer,engineer",
        "description": "An employee or team member in the organization",
        "id": "concept_person",
        "sample_values": "Alice Chen, Bob Martinez, Carol Johnson",
    },
    {
        "name": "Department",
        "nlp_terms": "department,departments,team,division,org unit,group",
        "description": "An organizational unit or department",
        "id": "concept_department",
        "sample_values": "Engineering, Marketing, Data Science, Product",
    },
    {
        "name": "Project",
        "nlp_terms": "project,projects,initiative,work,assignment,program",
        "description": "A project or work initiative",
        "id": "concept_project",
        "sample_values": "Project Phoenix, Customer Portal, Data Pipeline",
    },
    {
        "name": "Skill",
        "nlp_terms": "skill,skills,technology,tech,expertise,competency,ability",
        "description": "A technical skill or competency",
        "id": "concept_skill",
        "sample_values": "Python, Machine Learning, UI/UX Design",
    },
    {
        "name": "Location",
        "nlp_terms": "location,office,site,place,city,building,headquarters,HQ",
        "description": "A physical office location",
        "id": "concept_location",
        "sample_values": "San Francisco HQ, Austin Office, London Office",
    },
]

FULLTEXT_INDEXES = [
    ("personNameIndex", "Person", "name"),
    ("departmentNameIndex", "Department", "name"),
    ("projectNameIndex", "Project", "name"),
    ("skillNameIndex", "Skill", "name"),
    ("locationNameIndex", "Location", "name"),
]


# ── operations ────────────────────────────────────────────────

def clean(session):
    """Remove all sample data nodes and indexes."""
    print("Cleaning existing sample data...")

    # Drop fulltext indexes
    for idx_name, _, _ in FULLTEXT_INDEXES:
        try:
            _run(session, f"DROP INDEX {idx_name} IF EXISTS")
        except Exception:
            pass
    try:
        _run(session, "DROP INDEX globalSampleNameIndex IF EXISTS")
    except Exception:
        pass

    # Delete sample Concept nodes (by id prefix)
    _run(session, "MATCH (c:Concept) WHERE c.id STARTS WITH 'concept_' DETACH DELETE c")

    # Delete sample data nodes
    for label in ("Person", "Department", "Project", "Skill", "Location"):
        _run(session, f"MATCH (n:{label}) DETACH DELETE n")

    print("  Done.")


def create_nodes(session):
    """MERGE all nodes across 5 labels."""
    print("Creating nodes...")

    for p in PERSONS:
        _run(session,
             "MERGE (n:Person {name: $name}) "
             "SET n.title = $title, n.email = $email, n.hire_year = $hire_year",
             **p)

    for d in DEPARTMENTS:
        _run(session,
             "MERGE (n:Department {name: $name}) "
             "SET n.budget = $budget, n.floor = $floor",
             **d)

    for p in PROJECTS:
        _run(session,
             "MERGE (n:Project {name: $name}) "
             "SET n.status = $status, n.start_year = $start_year",
             **p)

    for s in SKILLS:
        _run(session,
             "MERGE (n:Skill {name: $name}) "
             "SET n.category = $category",
             **s)

    for loc in LOCATIONS:
        _run(session,
             "MERGE (n:Location {name: $name}) "
             "SET n.city = $city, n.country = $country",
             **loc)

    print(f"  {len(PERSONS)} Person, {len(DEPARTMENTS)} Department, "
          f"{len(PROJECTS)} Project, {len(SKILLS)} Skill, {len(LOCATIONS)} Location")


def create_relationships(session):
    """MERGE all relationships."""
    print("Creating relationships...")
    count = 0

    for person, dept in WORKS_IN:
        _run(session,
             "MATCH (p:Person {name: $person}) MATCH (d:Department {name: $dept}) "
             "MERGE (p)-[:WORKS_IN]->(d)",
             person=person, dept=dept)
        count += 1

    for person, project in WORKS_ON:
        _run(session,
             "MATCH (p:Person {name: $person}) MATCH (proj:Project {name: $project}) "
             "MERGE (p)-[:WORKS_ON]->(proj)",
             person=person, project=project)
        count += 1

    for person, skill in HAS_SKILL:
        _run(session,
             "MATCH (p:Person {name: $person}) MATCH (s:Skill {name: $skill}) "
             "MERGE (p)-[:HAS_SKILL]->(s)",
             person=person, skill=skill)
        count += 1

    for person, location in PERSON_LOCATED_IN:
        _run(session,
             "MATCH (p:Person {name: $person}) MATCH (l:Location {name: $location}) "
             "MERGE (p)-[:LOCATED_IN]->(l)",
             person=person, location=location)
        count += 1

    for dept, location in DEPT_LOCATED_IN:
        _run(session,
             "MATCH (d:Department {name: $dept}) MATCH (l:Location {name: $location}) "
             "MERGE (d)-[:LOCATED_IN]->(l)",
             dept=dept, location=location)
        count += 1

    for project, dept in OWNED_BY:
        _run(session,
             "MATCH (proj:Project {name: $project}) MATCH (d:Department {name: $dept}) "
             "MERGE (proj)-[:OWNED_BY]->(d)",
             project=project, dept=dept)
        count += 1

    for person, dept in MANAGES:
        _run(session,
             "MATCH (p:Person {name: $person}) MATCH (d:Department {name: $dept}) "
             "MERGE (p)-[:MANAGES]->(d)",
             person=person, dept=dept)
        count += 1

    for person, manager in REPORTS_TO:
        _run(session,
             "MATCH (p:Person {name: $person}) MATCH (mgr:Person {name: $manager}) "
             "MERGE (p)-[:REPORTS_TO]->(mgr)",
             person=person, manager=manager)
        count += 1

    print(f"  {count} relationships created.")


def create_concepts(session):
    """MERGE :Concept nodes for each label."""
    print("Creating Concept nodes...")

    for c in CONCEPTS:
        _run(session,
             "MERGE (n:Concept {name: $name}) "
             "SET n.nlp_terms = $nlp_terms, n.description = $description, "
             "n.id = $id, n.sample_values = $sample_values",
             **c)

    print(f"  {len(CONCEPTS)} Concept nodes.")


def create_indexes(session):
    """Create FULLTEXT indexes (per-label + one global)."""
    print("Creating FULLTEXT indexes...")

    for idx_name, label, prop in FULLTEXT_INDEXES:
        try:
            _run(session,
                 f"CREATE FULLTEXT INDEX {idx_name} IF NOT EXISTS "
                 f"FOR (n:{label}) ON EACH [n.{prop}]")
        except Exception as exc:
            print(f"  Warning: {idx_name}: {exc}")

    # Global index covering all 5 labels
    try:
        _run(session,
             "CREATE FULLTEXT INDEX globalSampleNameIndex IF NOT EXISTS "
             "FOR (n:Person|Department|Project|Skill|Location) ON EACH [n.name]")
    except Exception as exc:
        print(f"  Warning: globalSampleNameIndex: {exc}")

    print(f"  {len(FULLTEXT_INDEXES) + 1} indexes created. Waiting for population...")

    try:
        session.run("CALL db.awaitIndexes(300)").consume()
        print("  Indexes online.")
    except Exception as exc:
        print(f"  Warning: could not await indexes: {exc}")
        time.sleep(5)


def verify(session):
    """Print summary counts."""
    print("\n── Verification ──")

    for label in ("Person", "Department", "Project", "Skill", "Location", "Concept"):
        result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
        print(f"  {label}: {result['c']} nodes")

    result = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
    print(f"  Total relationships: {result['c']}")

    result = session.run(
        "SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT' RETURN count(*) AS c"
    ).single()
    print(f"  FULLTEXT indexes: {result['c']}")

    print("\nSample data ingestion complete!")


# ── main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest sample org-chart data into Neo4j")
    parser.add_argument("--clean", action="store_true", help="Drop and recreate all sample data")
    args = parser.parse_args()

    uri, user, password, database = _get_connection()
    print(f"Connecting to {uri} (database: {database})...")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("Connected.\n")

    with driver.session(database=database) as session:
        if args.clean:
            clean(session)
            print()

        create_nodes(session)
        create_relationships(session)
        create_concepts(session)
        create_indexes(session)
        verify(session)

    driver.close()


if __name__ == "__main__":
    main()
