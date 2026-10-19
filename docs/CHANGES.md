# Bugs fixed

| Area | Problem | Fix |
|---|---|---|
| Delete document | Only the legacy `pages` collection was cleared; `chunks_small`, `chunks_large` and the TF-IDF index kept the document, so it still appeared in the list and in answers | `indexer.delete_document` purges all three collections and the SQLite index |
| IDF | Only the new document's terms were refreshed, all other IDF values were stale | Whole vocabulary recomputed on add and delete |
| Greeting detection | `startswith` on a prefix list: "history of Rome" (hi), "okta SSO setup" (ok) were answered as small talk | `chitchat.py`, whole-word matching, whole message must be small talk |
| Chat history | Messages without a session were returned in EVERY session; limit returned the oldest N | Orphans adopted into one session at startup; strict filter; newest N |
| Streaming retry | Tokens of the failed first attempt stayed on screen and the retry was appended | Server sends `reset`; client clears |
| Not found | The question and result were not saved, so a reload lost them | Saved with a `notfound` role |
| Feedback link | Feedback matched logs by query text only | `query_id` returned and stored |
| Chat export | LLM and user text inserted into HTML unescaped (XSS) | `html.escape` |
| PDF export | Light text on a white page (unreadable); `<` in text crashed ReportLab; None scores crashed | Light palette, escaping, null-safe |
| Compare thresholds | Retrieval, generation and evaluation ran three times with identical results | Run once, vary only pass/fail |
| Snippet fallback | `get_doc_snippet` returned a placeholder, not text | Fallback uses real chunks |
| Multi-doc search | Used the legacy collection | Uses the large-chunk collection |
| LLM calls | Six separate Groq clients, no retry on HTTP 429 | `llm.py`, shared client, backoff |
| System prompt | Sent as a user message plus a fake assistant reply | Real `system` role; history filtered to user/assistant |
| Server | Debug reloader loaded models twice | `use_reloader=False` |
| Score trend | Showed the oldest 30 questions | Newest 30, chronological |


# Version 3: documents with tables, figures, formulas, spreadsheets and code

| Capability | How it works |
|---|---|
| PDF tables | PyMuPDF table detection; each table is kept as a Markdown chunk with its caption and page, and removed from the running prose |
| Figures | Embedded images and vector diagrams are saved with their caption and the sentence that cites them; the answer shows the figure when you name it, when it was evidence, or when the question is visual |
| Formulas, charts, scans | Optional "Deep analysis" sends pages and figures to a Groq vision model (formulas as LaTeX). Without it, text formulas are indexed as text only |
| Word, PowerPoint | Headings, tables, pictures, equations as text, slide text and speaker notes |
| Excel, CSV | Statistics chunk, row-group chunks, and exact pandas calculations for averages, counts and top-N |
| Code, notebooks, zip projects | Language-aware windows keep file name and line numbers; the prompt explains code and quotes it in fenced blocks |
| Citations | Every answer lists the passages used: document, page, table, code file and lines |
| Memory | You choose 2 to 12 recent turns; older turns are summarised; follow-ups are rewritten into standalone questions for searching |
| Chat | A separate direct conversation with the model, clearly marked as not from your documents and never logged to the analytics |


# Version 3.1

| Change | Detail |
|---|---|
| Deep analysis rate limits | A "request too large" error shrinks the request and then tries the next model; a 429 waits the number of seconds Groq asks for; one failed page no longer abandons the rest |
| Fewer vision calls | Only pages with figures, tables, formulas or little text go to the vision model (set VISION_ALL_PAGES=1 for all); captioned figures are not described separately |
| Regenerate | The new answer is stored as another version of the same answer (1/2, 2/2) in Ask and Chat; the question is saved once |
| Ask layout | Search scope and options are dropdowns; chats list and inspector can be hidden; the reading area is wider |
| Chat | Conversation list with search, pin, rename and delete (stored in this browser) |
| Accuracy | Insights shows measured accuracy with a one-click test and history; every benchmark run is saved |


# Version 3.2

| Change | Detail |
|---|---|
| Automatic summary, entities, tags | Created right after every upload (one model call over passages spread across the whole document) and shown in the document details; Regenerate is available |
| Answer rendering | `<br>` inside table cells becomes a real line break; numbered lists continue across blank lines; nested lists, quotes, rules, links, strikethrough, task lists and table alignment; evidence underlines no longer break list numbering |
| Consistent text width | Paragraphs, lists and tables share one width and use balanced line breaking |
| Figure requests | "Give me only the diagram" returns the figure itself with a one-line caption; figures are ranked by relevance so unrelated pictures are not shown; near-blank banner images are ignored |
| Printed page numbers | If a PDF prints its own page numbers ("Page | 14") the answer cites those |
| Delete questions | Hover a question for a trash button, or press Select to tick several questions and delete them with their answers; memory follows what is left |
| Smooth motion | Panels, dropdowns, dialogs, views and new answers animate (respects the reduced-motion setting) |
| Quiz me | Multiple-choice quiz with explanations and a review of mistakes, from any document |
