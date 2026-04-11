---
name: ask
description: "Query the knowledge base with a specific question. Searches across all ingested papers and wiki pages to produce a cited, evidence-based answer. Use for targeted questions about the literature."
argument-hint: "<question>"
---

# Ask the Knowledge Base

Question: **$ARGUMENTS**

## Workflow

1. **Parse the question.** Identify key entities: author names, method names, benchmark names, concepts.

2. **Search the wiki** for relevant content:
   - Grep `wiki/` for each key entity
   - Read `wiki/index.md` to identify relevant paper, topic, and method pages
   - Check the index for papers matching the query terms (by title, venue)

3. **Read relevant pages** (prioritized):
   - Topic pages matching the question's subject
   - Method pages if the question is about techniques
   - Benchmark pages if the question is about results/performance
   - Paper pages that discuss the relevant concepts
   - `wiki/overview.md` for high-level context

4. **Synthesize an answer** that:
   - Directly answers the question
   - Cites specific papers using `[[slug]]` wikilinks for every claim
   - Includes concrete numbers from results tables when relevant
   - Notes disagreements between papers if they exist
   - Flags when the knowledge base has insufficient coverage to fully answer
   - Suggests specific papers to `/ingest` if the answer would benefit from more coverage

5. **Format the response:**
   - Lead with a direct answer (1-3 sentences)
   - Follow with evidence and citations
   - End with confidence assessment: HIGH (multiple corroborating sources), MEDIUM (limited sources), LOW (extrapolating beyond what's ingested)

6. **Recommend next commands** — always end every response with a "What to do next:" block. Pick 2–3 commands that are the logical follow-on given the question and answer:
   - Confidence LOW or coverage thin → `/discover "{topic}"` or `/ingest <slug of relevant discovered paper>`
   - Two methods compared in the answer → `/compare <slug1> <slug2>`
   - A key paper was cited but not deeply explored → `/related <slug>`
   - Answer is complete and confident → `/gaps` to find what the knowledge base is still missing
   - Format:
     ```
     **What to do next:**
     - `/command arg` — reason
     - `/command arg` — reason
     ```

7. **Append to `wiki/log.md`:**
   ```bash
   echo "- [$(date "+%Y-%m-%d %H:%M")] **ask** -	\"{question}\" — answered from N sources, confidence: {level}" >> wiki/log.md
   ```
