"""Categories 11-14 — APOC Analysis, Text Search, Data Inspection, Collection Tools."""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


# ── Category 11: APOC Graph Analysis ─────────────────────────────────────────

async def apoc_nodes_degree(
    db: AbstractDatabase, *, label: str, limit: int = 25, **_: Any,
) -> list[dict]:
    """Return degree (total relationships) per node."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN n.name AS name, "
        f"apoc.node.degree(n) AS degree ORDER BY degree DESC LIMIT {int(limit)}"
    )


async def apoc_nodes_connected(
    db: AbstractDatabase, *, label_a: str, prop_a: str, val_a: str,
    label_b: str, prop_b: str, val_b: str, **_: Any,
) -> list[dict]:
    """Check if two nodes are connected."""
    return await db.execute_read(
        f"MATCH (a:`{label_a}` {{{prop_a}: $va}}), (b:`{label_b}` {{{prop_b}: $vb}}) "
        "RETURN EXISTS((a)-[*1..5]-(b)) AS connected",
        {"va": val_a, "vb": val_b},
    )


async def apoc_nodes_group(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    """Group nodes by a property value."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN n.`{prop}` AS value, count(n) AS cnt "
        f"ORDER BY cnt DESC LIMIT {int(limit)}"
    )


async def apoc_nodes_relationship_types(
    db: AbstractDatabase, *, label: str, limit: int = 25, **_: Any,
) -> list[dict]:
    """Return relationship types for nodes of a label."""
    return await db.execute_read(
        f"MATCH (n:`{label}`)-[r]-() "
        "RETURN DISTINCT type(r) AS rel_type, count(r) AS cnt "
        f"ORDER BY cnt DESC LIMIT {int(limit)}"
    )


async def apoc_nodes_links(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    limit: int = 25, **_: Any,
) -> list[dict]:
    """Return all links from a specific node."""
    return await db.execute_read(
        f"MATCH (n:`{label}` {{{prop}: $val}})-[r]-(m) "
        "RETURN type(r) AS rel, labels(m)[0] AS target_label, "
        f"properties(m) AS props LIMIT {int(limit)}",
        {"val": value},
    )


async def apoc_nodes_collapse(
    db: AbstractDatabase, *, label: str, group_prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    """Collapse nodes by property, showing group counts."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN n.`{group_prop}` AS group, "
        f"count(n) AS count, collect(n.name)[0..5] AS samples "
        f"ORDER BY count DESC LIMIT {int(limit)}"
    )


async def apoc_graph_fromCypher(
    db: AbstractDatabase, *, query: str, limit: int = 25, **_: Any,
) -> list[dict]:
    """Execute a Cypher query and return as graph structure."""
    return await db.execute_read(f"{query} LIMIT {int(limit)}")


async def apoc_graph_fromPaths(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    max_hops: int = 3, limit: int = 10, **_: Any,
) -> list[dict]:
    """Build graph from paths starting at a node."""
    return await db.execute_read(
        f"MATCH path = (n:`{label}` {{{prop}: $val}})-[*1..{max_hops}]-(m) "
        "WITH path LIMIT $lim "
        "RETURN [x IN nodes(path) | properties(x)] AS nodes, "
        "[r IN relationships(path) | type(r)] AS rels",
        {"val": value, "lim": limit},
    )


async def apoc_graph_fromData(
    db: AbstractDatabase, *, label: str, limit: int = 50, **_: Any,
) -> list[dict]:
    """Get graph data for a label."""
    return await db.execute_read(
        f"MATCH (n:`{label}`)-[r]->(m) "
        "RETURN properties(n) AS source, type(r) AS rel, properties(m) AS target "
        f"LIMIT {int(limit)}"
    )


async def apoc_graph_validate(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Validate graph consistency."""
    return await db.execute_read(
        "MATCH (n) WHERE size(labels(n)) = 0 RETURN count(n) AS unlabeled_nodes "
        "UNION ALL MATCH (n) WHERE size(keys(n)) = 0 RETURN count(n) AS empty_nodes"
    )


# ── Category 12: APOC Text Search ────────────────────────────────────────────

async def apoc_text_distance(
    db: AbstractDatabase, *, text1: str, text2: str, **_: Any,
) -> list[dict]:
    """Compute Levenshtein distance between two strings."""
    return await db.execute_read(
        "RETURN apoc.text.distance($t1, $t2) AS distance",
        {"t1": text1, "t2": text2},
    )


async def apoc_text_levenshteinDistance(
    db: AbstractDatabase, *, text1: str, text2: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "RETURN apoc.text.levenshteinDistance($t1, $t2) AS distance",
        {"t1": text1, "t2": text2},
    )


async def apoc_text_fuzzyMatch(
    db: AbstractDatabase, *, label: str, prop: str, search: str,
    threshold: float = 0.7, limit: int = 25, **_: Any,
) -> list[dict]:
    """Fuzzy match nodes by property using similarity."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) "
        f"WITH n, apoc.text.levenshteinSimilarity(toLower(n.`{prop}`), toLower($search)) AS sim "
        f"WHERE sim >= $threshold "
        f"RETURN properties(n) AS props, sim ORDER BY sim DESC LIMIT {int(limit)}",
        {"search": search, "threshold": threshold},
    )


async def apoc_text_similarity(
    db: AbstractDatabase, *, text1: str, text2: str, **_: Any,
) -> list[dict]:
    """Compute text similarity (0-1)."""
    return await db.execute_read(
        "RETURN apoc.text.levenshteinSimilarity($t1, $t2) AS similarity",
        {"t1": text1, "t2": text2},
    )


async def apoc_text_indexOf(
    db: AbstractDatabase, *, text: str, lookup: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "RETURN apoc.text.indexOf($text, $lookup) AS idx",
        {"text": text, "lookup": lookup},
    )


async def apoc_text_indexesOf(
    db: AbstractDatabase, *, text: str, lookup: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "RETURN apoc.text.indexesOf($text, $lookup) AS indexes",
        {"text": text, "lookup": lookup},
    )


async def apoc_text_replace(
    db: AbstractDatabase, *, text: str, regex: str, replacement: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "RETURN apoc.text.replace($text, $regex, $repl) AS result",
        {"text": text, "regex": regex, "repl": replacement},
    )


async def apoc_text_regreplace(
    db: AbstractDatabase, *, text: str, regex: str, replacement: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "RETURN apoc.text.regreplace($text, $regex, $repl) AS result",
        {"text": text, "regex": regex, "repl": replacement},
    )


async def apoc_text_split(
    db: AbstractDatabase, *, text: str, delimiter: str = ",", **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "RETURN apoc.text.split($text, $delim) AS parts",
        {"text": text, "delim": delimiter},
    )


async def apoc_text_join(
    db: AbstractDatabase, *, texts: list[str], delimiter: str = ",", **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "RETURN apoc.text.join($texts, $delim) AS result",
        {"texts": texts, "delim": delimiter},
    )


# ── Category 13: APOC Data Inspection (Map tools) ────────────────────────────

async def apoc_map_keys(
    db: AbstractDatabase, *, label: str, limit: int = 10, **_: Any,
) -> list[dict]:
    """Return all property keys for nodes of a label."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) WITH n LIMIT {int(limit)} "
        "RETURN keys(n) AS map_keys"
    )


async def apoc_map_values(
    db: AbstractDatabase, *, label: str, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WITH n LIMIT {int(limit)} "
        "RETURN [k IN keys(n) | n[k]] AS map_values"
    )


async def apoc_map_entries(
    db: AbstractDatabase, *, label: str, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WITH n LIMIT {int(limit)} "
        "RETURN [k IN keys(n) | {{key: k, value: n[k]}}] AS entries"
    )


async def apoc_map_fromPairs(
    db: AbstractDatabase, *, label: str, key_prop: str, value_prop: str,
    limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) "
        f"RETURN n.`{key_prop}` AS key, n.`{value_prop}` AS value "
        f"LIMIT {int(limit)}"
    )


async def apoc_map_subset(
    db: AbstractDatabase, *, label: str, props: str, limit: int = 25, **_: Any,
) -> list[dict]:
    prop_list = [p.strip() for p in props.split(",")]
    return_clause = ", ".join(f"n.`{p}` AS `{p}`" for p in prop_list)
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN {return_clause} LIMIT {int(limit)}"
    )


async def apoc_map_clean(
    db: AbstractDatabase, *, label: str, limit: int = 25, **_: Any,
) -> list[dict]:
    """Return nodes with null properties removed."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) WITH n LIMIT {int(limit)} "
        "RETURN [k IN keys(n) WHERE n[k] IS NOT NULL | [k, n[k]]] AS clean_props"
    )


async def apoc_map_merge(
    db: AbstractDatabase, *, label_a: str, label_b: str, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a:`{label_a}`), (b:`{label_b}`) "
        f"WITH a, b LIMIT {int(limit)} "
        "RETURN properties(a) AS props_a, properties(b) AS props_b"
    )


async def apoc_map_get(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL "
        f"RETURN n.`{prop}` AS value LIMIT {int(limit)}"
    )


# ── Category 14: APOC Collection Utilities ───────────────────────────────────

async def apoc_coll_contains(
    db: AbstractDatabase, *, label: str, prop: str, value: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE $val IN n.`{prop}` "
        f"RETURN properties(n) AS props LIMIT {int(limit)}",
        {"val": value},
    )


async def apoc_coll_intersection(
    db: AbstractDatabase, *, label: str, prop: str, values: list[str], **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) "
        f"RETURN n.name AS name, apoc.coll.intersection(n.`{prop}`, $vals) AS common",
        {"vals": values},
    )


async def apoc_coll_union(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WITH collect(n.`{prop}`) AS all_vals "
        "RETURN apoc.coll.union(all_vals, []) AS unique_values"
    )


async def apoc_coll_flatten(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WITH collect(n.`{prop}`) AS nested "
        "RETURN apoc.coll.flatten(nested) AS flat"
    )


async def apoc_coll_sort(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN n.`{prop}` AS value "
        f"ORDER BY n.`{prop}` LIMIT {int(limit)}"
    )


async def apoc_coll_frequencies(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN n.`{prop}` AS value, count(*) AS frequency "
        f"ORDER BY frequency DESC LIMIT {int(limit)}"
    )


async def apoc_coll_duplicates(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WITH n.`{prop}` AS val, count(*) AS cnt "
        f"WHERE cnt > 1 RETURN val, cnt ORDER BY cnt DESC LIMIT {int(limit)}"
    )


async def apoc_coll_difference(
    db: AbstractDatabase, *, label: str, prop: str, values: list[str], **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) "
        f"RETURN n.name AS name, apoc.coll.subtract(n.`{prop}`, $vals) AS diff",
        {"vals": values},
    )


async def apoc_coll_reverse(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN n.`{prop}` AS value "
        f"ORDER BY n.`{prop}` DESC LIMIT {int(limit)}"
    )


async def apoc_coll_partition(
    db: AbstractDatabase, *, label: str, prop: str, size: int = 5, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WITH collect(n.`{prop}`) AS all_vals "
        f"RETURN apoc.coll.partition(all_vals, {int(size)}) AS partitions"
    )


# ── Registries ───────────────────────────────────────────────────────────────

APOC_ANALYSIS_TOOLS: dict = {
    "apoc_nodes_degree": apoc_nodes_degree,
    "apoc_nodes_connected": apoc_nodes_connected,
    "apoc_nodes_group": apoc_nodes_group,
    "apoc_nodes_relationship_types": apoc_nodes_relationship_types,
    "apoc_nodes_links": apoc_nodes_links,
    "apoc_nodes_collapse": apoc_nodes_collapse,
    "apoc_graph_fromCypher": apoc_graph_fromCypher,
    "apoc_graph_fromPaths": apoc_graph_fromPaths,
    "apoc_graph_fromData": apoc_graph_fromData,
    "apoc_graph_validate": apoc_graph_validate,
}

APOC_TEXT_TOOLS: dict = {
    "apoc_text_indexOf": apoc_text_indexOf,
    "apoc_text_indexesOf": apoc_text_indexesOf,
    "apoc_text_replace": apoc_text_replace,
    "apoc_text_regreplace": apoc_text_regreplace,
    "apoc_text_split": apoc_text_split,
    "apoc_text_join": apoc_text_join,
    "apoc_text_distance": apoc_text_distance,
    "apoc_text_levenshteinDistance": apoc_text_levenshteinDistance,
    "apoc_text_fuzzyMatch": apoc_text_fuzzyMatch,
    "apoc_text_similarity": apoc_text_similarity,
}

APOC_MAP_TOOLS: dict = {
    "apoc_map_keys": apoc_map_keys,
    "apoc_map_values": apoc_map_values,
    "apoc_map_entries": apoc_map_entries,
    "apoc_map_fromPairs": apoc_map_fromPairs,
    "apoc_map_subset": apoc_map_subset,
    "apoc_map_clean": apoc_map_clean,
    "apoc_map_merge": apoc_map_merge,
    "apoc_map_get": apoc_map_get,
}

APOC_COLLECTION_TOOLS: dict = {
    "apoc_coll_contains": apoc_coll_contains,
    "apoc_coll_intersection": apoc_coll_intersection,
    "apoc_coll_union": apoc_coll_union,
    "apoc_coll_difference": apoc_coll_difference,
    "apoc_coll_flatten": apoc_coll_flatten,
    "apoc_coll_sort": apoc_coll_sort,
    "apoc_coll_reverse": apoc_coll_reverse,
    "apoc_coll_partition": apoc_coll_partition,
    "apoc_coll_frequencies": apoc_coll_frequencies,
    "apoc_coll_duplicates": apoc_coll_duplicates,
}
