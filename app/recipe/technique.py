# LLM + RAG outputs technique Plan JSON 
from rag import generate, query, build_context
from config import OPEN_API_KEY, LLM_MODEL
import json
import chromadb
from config import get_collection
from pathlib import Path 
from openai import OpenAI 
from config import OPEN_API_KEY, LLM_MODEL
from collections import deque

BASE_DIR = Path(__file__).resolve().parent.parent.parent
chroma_client = chromadb.PersistentClient(BASE_DIR / "data" / "vectorstore")
collection = get_collection(chroma_client)


TECHNIQUE_PLAN_SYSTEM = '''You are a culinary technique decision engine.

You will receive a JSON payload containing:
1) A decision_key (e.g., "method_selection", "sauce_architecture")
2) A normalized_request describing the user's intent and constraints
3) Three stable_questions that define what must be decided for this decision_key
4) A list of evidence chunks retrieved from authoritative sources
   (each chunk has: source, chunk_id, and text)

Your task:
Produce a Technique Plan decision for the given decision_key.

You must:
- Use the evidence chunks as grounding for your decisions
- Respect all constraints and priorities in normalized_request
- Answer the stable_questions implicitly through your final decision
- Synthesize a single coherent decision, not three separate answers
- Make the decision actionable, concrete, and step-specific when possible
  (include techniques, timing, ingredient handling, or sequencing)

STRICT RULES:
- Do NOT invent facts, techniques, or sources
- Do NOT cite sources that are not present in the provided evidence list
- Do NOT hallucinate chunk_ids
- Citations must directly support the decision or explanation
- Prefer fewer, stronger citations (1-3 is ideal)
- If evidence is insufficient or indirect:
  - Still make a best-effort decision
  - Set confidence to "low"
  - Set needs_more_evidence to true
- If evidence clearly supports the decision:
  - Set confidence to "high" or "medium"
  - Set needs_more_evidence to false

Decision quality priorities (in order):
1) Hard constraints (equipment, dietary, safety)
2) Intent (e.g., quick, elegant, creative, comforting)
3) Context (e.g., low stress, romantic, casual)
4) Culinary best practice derived from evidence

OUTPUT FORMAT (STRICT JSON ONLY):
Return a single JSON object with this exact structure:

{
  "<decision_key>": {
    "decision": string,
    "why": string,
    "confidence": "high" | "medium" | "low",
    "citations": [
      { "source": string, "chunk_id": string }
    ],
    "needs_more_evidence": boolean
  }
}

RECOMMENDATIONS FOR HIGH-QUALITY DECISIONS:
- Include specific preparation steps, temperatures, and timings where supported by evidence
- Mention ingredient handling or technique nuances that affect final quality
- Describe the reasoning behind your choices in the "why" section, linking them to intent and context
- If multiple techniques are possible, select the one that maximizes flavor, texture, and presentation while respecting constraints
- Focus on a single coherent approach rather than listing multiple options separately

- The top-level key MUST exactly match the provided decision_key
- Do NOT include markdown
- Do NOT include explanations outside the JSON
- Do NOT include additional keys
'''
# each: {"source":..., "chunk_id":..., "text":...}
# rag quries 
def query_evidence(questions, collection, top_k=3):
    evidence = []
    print("-------------------------")
    print(questions)

    for q in questions:
        results = query(q, collection)
        print("---------------------------")
        print(results)
        print("___________________________________")

        # make sure results exist
        if not results or not results["documents"] or not results["metadatas"]:
            continue

        docs_list = results["documents"][0]
        print("_______________________")
        print(docs_list)
        metas_list = results["metadatas"][0]
        print("______________________")
        print(metas_list)
        print("_____________________________")


        for i in range(len(docs_list)):
            doc_text = docs_list[i]
            meta = metas_list[i]

            # fallback inline without .get()
            source = meta["book"] if "book" in meta else (meta["source"] if "source" in meta else "unknown")
            doc_id = meta["doc_id"] if "doc_id" in meta else "0"
            page = meta["page"] if "page" in meta else "0"
            chunk = meta["chunk"] if "chunk" in meta else "0"

            evidence.append({
                "source": source,
                "chunk_id": f"{doc_id}_p{page}_c{chunk}",
                "text": doc_text
            })

    return evidence
       
# execute all of the questions of the model at the same time 
def technique_LLM(
    *,
    model: str,
    system_prompt: str,
    decision_key: str,
    normalized_request: dict,
    stable_questions: list[str],
    evidence_chunks: list[dict],
) -> dict:

    client = OpenAI()

    payload = {
        "decision_key": decision_key,
        "normalized_request": normalized_request,
        "stable_questions": stable_questions,
        "evidence": evidence_chunks
    }

    user_prompt = json.dumps(payload, ensure_ascii=False)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON safely
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end+1])
        raise




    # the queue is going to be steps which contains 
    '''
    decision key 
    slot 0,1,2
    stable_question for the LLM to answer it 
    rag_queries for the RAG to answer thsi 
    '''

import json
from datetime import datetime
from pathlib import Path

def save_technique_plan(steps, normalized_request, collection, output_dir="technique_plans"):
    """
    Executes all steps, stores a full Technique Plan with evidence and metadata,
    and saves it as a timestamped JSON file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Full plan dictionary
    full_plan = {
        "timestamp": datetime.now().isoformat(),
        "normalized_request": normalized_request,
        "plan": {},       # decisions keyed by decision_key
        "evidence": {}    # evidence keyed by decision_key
    }

    for step in steps:
        decision_key = step["decision_key"]
        stable_questions = step["stable_questions"]
        rag_queries = step["rag_queries"]

        # Retrieve evidence
        evidence_chunks = query_evidence(rag_queries, collection)

        # Store evidence for traceability
        full_plan["evidence"][decision_key] = evidence_chunks

        # Call LLM for the decision
        decision = technique_LLM(
            model=LLM_MODEL,
            system_prompt=TECHNIQUE_PLAN_SYSTEM,
            decision_key=decision_key,
            normalized_request=normalized_request,
            stable_questions=stable_questions,
            evidence_chunks=evidence_chunks
        )

        # Store decision keyed by decision_key
        full_plan["plan"][decision_key] = decision[decision_key]

    # Save to JSON file with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"technique_plan_{timestamp_str}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(full_plan, f, ensure_ascii=False, indent=2)

    print(f"Technique Plan saved to: {file_path}")
    return full_plan
    
