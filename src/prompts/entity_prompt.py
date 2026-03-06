ENTITY_EXTRACTION_PROMPT = """You are an entity extraction specialist. Given a user question and database context, extract all specific names, IDs, or values the user is searching for.

=== DATABASE CONTEXT ===
{label_descriptions}

=== USER QUESTION ===
"{question}"

=== INSTRUCTIONS ===
1. Identify any specific names, identifiers, or values mentioned in the question.
2. For each entity, determine:
   - value: the EXACT text from the question AS WRITTEN (preserve typos, misspellings, and original casing)
   - type: "name" (a name/title), "id" (an identifier/code), or "keyword" (a general search term)
   - likely_label: which database label this entity most likely belongs to (from the context above), or null if unknown
3. Do NOT extract generic words like "all", "list", "show", "how many" — only specific search values.
4. Do NOT correct spellings or typos — copy the exact text as it appears.
5. If the question has no specific entities (e.g., "How many jobs are there?"), return an empty list.

=== OUTPUT FORMAT ===
Return ONLY valid JSON, nothing else:
{{"entities": [{{"value": "...", "type": "name|id|keyword", "likely_label": "LabelName or null"}}]}}

If no entities found:
{{"entities": []}}"""
