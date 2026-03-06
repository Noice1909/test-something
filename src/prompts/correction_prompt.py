CORRECTION_PROMPT = """The following Cypher query failed validation. Fix it.

=== FAILED QUERY ===
{failed_cypher}

=== ERROR ===
{error_message}

=== ORIGINAL QUESTION ===
{question}

=== RESOLVED ENTITIES ===
{mapped_entities}
(CRITICAL: Use these exact resolved values in your WHERE clauses.
These entities have been verified against the database through fuzzy matching.)

=== DATABASE SCHEMA ===
{filtered_schema}

=== COMMON FIXES ===
- "Unknown label": Check the exact label spelling in the schema above.
- "Unknown relationship type": Check exact relationship type name and direction.
- "Unknown property": Check the exact property names listed per label in the schema.
  Different labels may have different property names (e.g., "App_Name" vs "name").
- "wrong direction": Reverse the relationship arrow to match the schema.
- "Syntax error": Check for missing parentheses, brackets, or typos.
- "Missing RETURN": Every query needs a RETURN clause.

=== INSTRUCTIONS ===
1. Use the RESOLVED ENTITIES values (not the original question text) for entity filters.
2. Fix the query based on the error and schema.
3. Return ONLY the corrected Cypher query.

No explanations, no markdown, no backticks."""
