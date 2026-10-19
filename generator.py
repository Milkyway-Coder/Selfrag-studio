"""
generator.py — LLM generation with improved prompts.

Key prompt improvements over previous version:
1. Explicitly instructs the LLM to read ALL chunks before answering
2. Asks the LLM to synthesize across chunks, not just quote the first one
3. Requests structured answers for complex questions
4. Clearer "not found" instructions to reduce hallucination
5. Conversation history injected properly for multi-turn memory
"""

import re
from llm import client, chat, chat_text, GEN_MODEL   # `client` re-exported for app.py

MEMORY_TURNS = 4   # past exchanges to include


# ── System prompt ─────────────────────────────────────────────────────────────

def _system_prompt(context):
    return f"""You are a precise document assistant. Your job is to answer questions using ONLY the context passages provided below.

IMPORTANT RULES:
1. Read ALL context passages carefully before writing your answer — the answer may be split across multiple passages.
2. Synthesize information from multiple passages when needed — do not just quote one passage and ignore others.
3. If the context contains a direct answer, give it clearly and completely.
4. If the context contains partial information, give what is available and note what is missing.
5. If the context does not contain the answer at all, say exactly: "This information is not available in the indexed documents."
6. Never invent facts. Never use knowledge from outside the context.
7. For technical or factual questions, be specific — include numbers, names, dates, and terms exactly as they appear in the context.
8. Keep your answer focused. Do not pad with unnecessary introductions.
9. Passages may contain Markdown tables, source code in fenced blocks, LaTeX formulas, spreadsheet statistics and figure descriptions. Reproduce the table rows that matter exactly; quote code exactly inside fenced blocks with the language name; write formulas in LaTeX ($...$ inline, $$...$$ for display); when a figure is relevant, refer to it by its label (for example "Figure 3") and say what it shows.
10. A passage that begins "Computed result" holds exact numbers calculated from the full spreadsheet. Prefer it over any estimate and state the numbers as given.
11. For questions about code in the passages, explain what the code does, name the file and line range, and say so if the needed code is not in the passages.
12. Write clean Markdown only: never HTML tags such as <br>. Inside a table cell, separate items with semicolons. Keep list items short and parallel, and put a blank line before and after tables and lists.
13. Never mention "Passage 1", "Passage 2" or similar labels in your answer. Refer to the page, figure, table or file instead.

CONTEXT PASSAGES:
{context}

---
Answer the user's question using the context above."""


def _system_prompt_multidoc(context):
    return f"""You are a precise document assistant searching across multiple source documents. Answer using ONLY the context passages provided.

IMPORTANT RULES:
1. Read ALL passages carefully — information may be spread across multiple documents.
2. After every factual claim, cite the source document in square brackets like this: [doc-name]
3. If different documents say different things about the same topic, note the difference.
4. If the answer is not in any document, say: "This information is not available in the indexed documents."
5. Never invent facts. Never use outside knowledge.
6. Be specific — include exact terms, numbers, and definitions from the source text.

SOURCE PASSAGES:
{context}

---
Answer with inline citations [doc-name] after each claim."""


# ── Standard blocking generation ──────────────────────────────────────────────

def generate(query, context_chunks):
    """Single-turn, blocking generation."""
    if not context_chunks:
        raise ValueError("No context chunks provided.")

    context  = _format_context(context_chunks)
    messages = _build_messages(query, context, history=None, multi_doc=False)

    print(f"\nGenerating answer for: '{query[:60]}...'")
    try:
        answer = chat_text(messages, model=GEN_MODEL,
                           temperature=0.15,   # lower = more factual, less creative
                           timeout=45)
        if not answer:
            raise ValueError("LLM returned empty response.")
        print(f"Answer ({len(answer.split())} words): {answer[:100]}...")
        return answer
    except Exception as e:
        print(f"Generator error: {e}"); raise


# ── Streaming generation ──────────────────────────────────────────────────────

def generate_stream(query, context_chunks, history=None, labels=None, memory_turns=MEMORY_TURNS, summary=None):
    """
    Streams answer tokens one at a time for the chat UI.
    Includes conversation history for multi-turn memory.
    """
    if not context_chunks:
        yield "This information is not available in the indexed documents."
        return

    context  = _format_context(context_chunks, labels)
    messages = _build_messages(query, context, history, multi_doc=False,
                               memory_turns=memory_turns, summary=summary)

    print(f"\nStreaming answer for: '{query[:60]}...' "
          f"(memory: {len(history or [])} msgs)")
    try:
        stream = chat(messages, model=GEN_MODEL, temperature=0.15,
                      timeout=60, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as e:
        print(f"Streaming error: {e}")
        yield f"\n[Generation error: {e}]"


# ── Multi-doc with citations ──────────────────────────────────────────────────

def generate_with_citations(query, labeled_chunks, history=None):
    """
    Multi-doc generation — instructs the LLM to cite every claim.

    labeled_chunks: list of {"doc_id": str, "text": str}
    Returns: {"answer": str, "cited_sources": [str]}
    """
    if not labeled_chunks:
        raise ValueError("No chunks provided.")

    # Format context with clear source labels
    context_parts = []
    for c in labeled_chunks:
        context_parts.append(f"[Source: {c['doc_id']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    system  = _system_prompt_multidoc(context)
    messages = [{"role": "system", "content": system}]

    messages.extend(_clean_history(history))
    messages.append({"role": "user", "content": query})

    print(f"\nGenerating multi-doc answer for: '{query[:60]}...'")
    try:
        answer = chat_text(messages, model=GEN_MODEL, temperature=0.15, timeout=50)
        if not answer:
            raise ValueError("Empty response from LLM.")

        valid_ids     = {c["doc_id"] for c in labeled_chunks}
        cited_sources = list(set(m for m in re.findall(r'\[([^\]]+)\]', answer)
                                 if m in valid_ids))
        print(f"Multi-doc answer: {answer[:100]}...")
        print(f"Cited: {cited_sources}")
        return {"answer": answer, "cited_sources": cited_sources}
    except Exception as e:
        print(f"Multi-doc error: {e}"); raise


# ── Query reformulation ───────────────────────────────────────────────────────

def reformulate_query(original_query, failure_reason="low score"):
    """
    Rewrites a failed query to be more retrieval-friendly.
    Returns the new query string, or None if unchanged/failed.
    """
    prompt = (
        f'A document search query failed to find good results.\n'
        f'Original query: "{original_query}"\n'
        f'Failure reason: {failure_reason}\n\n'
        f'Rewrite the query as a more specific, keyword-focused search phrase '
        f'that would better match relevant passages in a technical document.\n'
        f'Return ONLY the rewritten query. No explanation. No quotes.'
    )
    try:
        new_q = chat_text([{"role": "user", "content": prompt}],
                          model=GEN_MODEL, temperature=0.3, timeout=15, retries=2)
        new_q = new_q.strip('"\'')
        if not new_q or new_q.lower() == original_query.lower():
            return None
        print(f"Reformulated: '{original_query}' → '{new_q}'")
        return new_q
    except Exception as e:
        print(f"Reformulation error: {e}"); return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(chunks, labels=None):
    """
    Formats retrieved chunks for the prompt. Each passage is numbered and, when known, labelled with
    its type and location (table, figure, code, page) so the model can cite it properly.
    """
    numbered = []
    for i, ch in enumerate(chunks, 1):
        lab = labels[i - 1] if labels and i - 1 < len(labels) and labels[i - 1] else ""
        numbered.append(f"[Passage {i}{' | ' + lab if lab else ''}]\n{ch.strip()}")
    return "\n\n---\n\n".join(numbered)


def _build_messages(query, context, history, multi_doc, memory_turns=MEMORY_TURNS, summary=None):
    """Builds the messages array with system prompt + history + current query."""
    sys_prompt = (_system_prompt_multidoc(context) if multi_doc
                  else _system_prompt(context))
    if summary:
        sys_prompt += ("\n\nSUMMARY OF THE EARLIER PART OF THIS CONVERSATION (for resolving references such as "
                       f"'it' or 'the second one'; it is not a source of facts):\n{summary}")

    messages = [{"role": "system", "content": sys_prompt}]

    # inject conversation history for multi-turn memory
    if history:
        messages.extend(_clean_history(history, memory_turns))

    messages.append({"role": "user", "content": query})
    return messages


def _clean_history(history, turns=MEMORY_TURNS, max_chars=1500):
    """Keeps only valid user/assistant turns (the chat log also stores greeting/fallback roles)."""
    ok = [t for t in (history or [])
          if isinstance(t, dict) and t.get("role") in ("user", "assistant")
          and str(t.get("content", "")).strip()]
    return [{"role": t["role"], "content": str(t["content"])[:max_chars]}
            for t in ok[-(turns * 2):]]