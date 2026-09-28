# Building an Agent That Learns from Every Interaction with Hindsight

Debugging production outages at 3 AM is fundamentally an information retrieval problem. Your primary API gateway returns HTTP 504 Gateway Timeouts, downstream worker pods crash from memory exhaustion, and your team scrambles across Slack threads, terminal buffers, and unindexed post-mortems trying to figure out what changed. Deep down, you know someone on your team fixed this exact issue six months ago—perhaps a leaked database connection pool session or an unindexed query scanning millions of rows—but that knowledge is buried in closed tickets or isolated inside the head of an off-shift engineer.

This phenomenon is **organizational memory loss**, and it costs engineering teams hundreds of hours during critical outages. When I built **MemoryOps**, an AI-powered incident response platform, my goal was to eliminate this wasted effort: create an SRE copilot that retains every resolved incident and recalls those exact learnings when new failures occur.

However, standard LLM architectures break down here. Passing raw, unstructured logs directly into an LLM context window is expensive, slow, and prone to hallucinating generic or dangerous remediation advice. To solve this, I built MemoryOps around persistent vector banks using [Hindsight](https://github.com/vectorize-io/hindsight), an open-source memory engine designed for autonomous agents. By pairing Hindsight's semantic memory retention and retrieval with Groq's high-speed inference engine, MemoryOps builds a stateful agent that grounds every recommendation in real historical engineering precedents.

In this article, I will explain how MemoryOps works under the hood, walk through the exact code for memory retention and recall, demonstrate a real incident workflow, and share key engineering lessons learned from integrating persistent memory into production systems.

---

## System Architecture: Decoupling Memory from Reasoning

Traditional LLM workflows treat models as stateless processors. You feed them prompt context, get a response, and discard the state. MemoryOps decouples **long-term memory storage** from **LLM reasoning execution**, treating memory as a persistent first-class data layer.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (React 18 + Vite)                            │
│  • Dashboard.jsx            • IncidentInvestigation.jsx                         │
│  • MemoryExplorer.jsx       • CreateIncident.jsx                                │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ HTTP / JSON REST
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            BACKEND (FastAPI + Python 3.12)                      │
│  • app/routers/incidents.py: Incident lifecycle & investigation pipeline       │
│  • app/hindsight_service.py: Async Hindsight SDK client wrapper (RETAIN/RECALL)│
│  • app/ai_service.py: Groq SDK client wrapper & JSON prompt enforcer           │
│  • app/models.py & schemas.py: SQLAlchemy ORM & Pydantic v2 schemas             │
└───────────────────────┬───────────────────────────────┬─────────────────────────┘
                        │                               │
                        ▼                               ▼
       ┌────────────────────────────────┐     ┌──────────────────────────────────┐
       │   RELATIONAL DB (SQLite)       │     │   HINDSIGHT MEMORY ENGINE        │
       │   • File: data/incidentiq.db   │     │   • SDK: hindsight-client        │
       │   • Model: Incident            │     │   • Operations: RETAIN & RECALL  │
       │   • Flag: memory_retained      │     │   • Persistent memory banks      │
       └────────────────────────────────┘     └────────────────┬─────────────────┘
                                                               │
                                                               ▼
                                                      ┌──────────────────────────┐
                                                      │  GROQ LLM REASONING      │
                                                      │  • Model: gpt-oss-20b    │
                                                      │  • Grounded Prompting    │
                                                      └──────────────────────────┘
```

The system operates across three core layers:

1. **Relational Storage (SQLite + SQLAlchemy)**: Manages core operational records in `data/incidentiq.db`. The `Incident` schema tracks service names, error signatures, symptoms, severity, root causes, resolution steps, timestamps, and a boolean flag (`memory_retained`) to prevent duplicate indexing.
2. **Persistent Memory Store (Hindsight Engine)**: Manages [Vectorize agent memory](https://vectorize.io/what-is-agent-memory) across isolated memory banks (`HINDSIGHT_BANK_ID="incidentiq"`). When an engineer resolves an outage, MemoryOps formats a structured experience document and executes a Hindsight `RETAIN` operation. When a new incident occurs, MemoryOps executes a semantic `RECALL` query to retrieve relevant historical resolutions.
3. **Reasoning Engine (Groq LLM)**: Injects recalled Hindsight memories into system prompts as grounded evidence, generating structured JSON recommendations while strictly enforcing evidence attribution.

---

## Code Deep-Dive: Storage and Retrieval Workflows

MemoryOps interacts with Hindsight via the official `hindsight-client` Python SDK in `backend/app/hindsight_service.py`.

### 1. Retaining Incident Experience (RETAIN)

When an engineer completes an incident investigation and marks it resolved, MemoryOps formats a structured experience document containing the service name, error signature, symptoms, root cause, and successful resolution. This experience is indexed into Hindsight via `aretain_incident`:

```python
async def aretain_incident(
    self,
    incident_id: str,
    service: str,
    error: str,
    symptoms: str,
    severity: str,
    root_cause: Optional[str] = None,
    resolution: Optional[str] = None,
    outcome: str = "Resolved",
) -> Dict[str, Any]:
    content_lines = [
        f"Incident ID: {incident_id}",
        f"Service: {service}",
        f"Error: {error}",
        f"Symptoms: {symptoms}",
        f"Severity: {severity}",
        f"Outcome: {outcome}",
    ]
    if root_cause:
        content_lines.append(f"Root Cause: {root_cause}")
    if resolution:
        content_lines.append(f"Resolution: {resolution}")

    content_text = "\n".join(content_lines)
    metadata = {"incident_id": incident_id, "service": service, "severity": severity, "outcome": outcome}

    client = self.get_client()
    try:
        response = await client.aretain(
            bank_id=self.bank_id,
            content=content_text,
            metadata=metadata,
            document_id=incident_id,
            tags=[service, severity, outcome],
        )
        return {"success": True, "incident_id": incident_id, "bank_id": self.bank_id}
    finally:
        if self._client is None:
            await client.aclose()
```

By passing explicit tags (`[service, severity, outcome]`) alongside structured content, Hindsight enables filtered metadata queries alongside semantic vector search.

### 2. Semantic Memory Retrieval (RECALL)

During an active incident investigation (`POST /api/v1/incidents/{incident_id}/analyze`), MemoryOps formulates a search query combining the affected service, error message, and observed symptoms, then invokes `arecall_memories`:

```python
async def arecall_memories(
    self,
    query: str,
    budget: str = "mid",
    max_tokens: int = 4096,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    client = self.get_client()
    try:
        response = await client.arecall(
            bank_id=self.bank_id,
            query=query,
            budget=budget,
            max_tokens=max_tokens,
            tags=tags,
        )
        return {"success": True, "query": query, "bank_id": self.bank_id, "results": response}
    except Exception as e:
        logger.warning(f"Failed to recall memories from Hindsight: {e}")
        return {"success": False, "query": query, "results": None, "error": str(e)}
    finally:
        if self._client is None:
            await client.aclose()
```

Hindsight evaluates the query against the memory bank using semantic vector similarity, surfacing past incidents with similar failure patterns even when exact keywords or service names differ.

### 3. Anti-Hallucination Prompting & Response Enforcement

Once historical memories are retrieved, they are passed into `AIIncidentService.aanalyze_incident` in `backend/app/ai_service.py`. The system prompt strictly separates historical evidence from LLM reasoning and forces a structured JSON schema:

```python
SYSTEM_PROMPT = """You are MemoryOps AI, an expert SRE/DevOps incident response assistant.
Your task is to analyze an incoming IT/DevOps incident using recalled historical incident memories from Hindsight.

CRITICAL INSTRUCTIONS:
1. Distinguish clearly between historical evidence (retrieved from Hindsight memories) and your own AI analysis/reasoning.
2. NEVER invent or fabricate historical incidents or incident IDs. Only reference historical incidents that are explicitly present in the provided RECALLED HISTORICAL MEMORIES.
3. If no relevant historical incidents exist or match, explicitly state that no historical incidents were found in Hindsight and base your analysis solely on general DevOps best practices.
4. Return your output STRICTLY as a valid JSON object matching this exact schema:
{
  "probable_root_cause": "Detailed explanation of the probable root cause",
  "recommended_action": "Specific step-by-step remediation or investigation steps",
  "confidence": "high" | "medium" | "low",
  "reasoning": "Clear explanation distinguishing historical evidence from general model reasoning",
  "supporting_historical_incidents": ["INC-XXX: Summary of relevant historical incident"]
}
"""
```

---

## Concrete Execution Workflow: Connection Pool Exhaustion

To see how persistent memory changes incident response behavior, consider a scenario backed by seed data in `backend/app/seed.py`.

1. **Past Outage (`INC-101`)**:
   - **Service**: Payment API
   - **Error**: Database connection timeout
   - **Symptoms**: High HTTP 504 Gateway Timeouts on `/v1/charge`, elevated API latency.
   - **Root Cause**: Connection pool exhaustion due to leaked unclosed DB sessions during a traffic spike.
   - **Resolution**: Increased connection pool size from 20 to 100 and deployed hotfix for session leak.
   - **Outcome**: Resolved and retained in Hindsight.

2. **New Outage (`INC-106`)**:
   - Six weeks later, an SRE receives a critical alert on the Customer Relational Database: `"FATAL: remaining connection slots are reserved for non-replication superuser connections"`.
   - The SRE opens MemoryOps and triggers an investigation (`POST /api/v1/incidents/INC-106/analyze`).

3. **Hindsight Memory Recall & Grounded Reasoning**:
   - MemoryOps executes a semantic `arecall_memories` query: `Service: Customer Relational Database | Error: Database connection problems | Symptoms: FATAL connection slots reserved`.
   - Hindsight matches the semantic signature of connection slot exhaustion against `INC-101`.
   - Groq AI receives `INC-101` as grounded context and generates an actionable recommendation: inspect active worker connections and configure idle connection timeouts in PgBouncer.
   - The UI displays the recommendation along with explicit evidence attribution:
     `"Supporting Historical Evidence: INC-101 - Database connection pool exhaustion due to leaked sessions"`.

4. **Human Verification and Memory Retention**:
   - The SRE verifies active connection counts, applies the PgBouncer configuration update, and confirms service recovery.
   - Clicking **"Resolve & Retain"** triggers `aretain_incident()`, adding `INC-106` to Hindsight and expanding the team's collective memory bank for future incidents.

---

## Key Engineering Lessons Learned

Building MemoryOps highlighted several technical lessons regarding agent memory integration:

### 1. Async Event Loop Safety in FastAPI
Invoking synchronous wrappers that call `asyncio.run()` inside an active FastAPI event loop causes runtime collisions (`RuntimeError: This event loop is already running`). To ensure thread safety and non-blocking performance, `hindsight_service.py` implements native async SDK calls (`aretain`, `arecall`, `areflect`) and awaits them cleanly inside router handlers.

### 2. Resilience to External Service Exhaustion
External APIs can encounter rate limits or credit exhaustion (`HTTP 402 Payment Required`). In `hindsight_service.py`, I implemented `is_insufficient_credits_error()` to detect 402 exceptions. When Hindsight Cloud returns a 402 response, MemoryOps catches the exception gracefully and falls back to querying previously resolved incidents directly from SQLite (`db.query(Incident).filter(Incident.outcome.ilike("resolved"))`). This ensures the SRE investigation workspace remains functional even when external memory endpoints are offline.

### 3. De-duplication with ORM Retention Flags
Re-indexing an incident into Hindsight every time an engineer edits incident notes pollutes the vector bank with duplicate memories. To prevent this, the `Incident` database model includes a boolean `memory_retained` flag. Retention executes only once when an incident transitions to `"Resolved"` with complete root cause data, setting `memory_retained = True` to block duplicate indexing.

### 4. Structuring Prompts for Evidence Attribution
LLMs generate compelling text, but in engineering workflows, plausible-sounding guesses are dangerous. Requiring structured JSON responses with mandatory `supporting_historical_incidents` arrays forces the model to cite specific historical incident IDs (`INC-101`), transforming generic AI chat into verifiable engineering analysis.

---

## Current Limitations and Planned Evolution

While MemoryOps provides an effective memory-augmented workflow, current limitations include:
- **Vector Service Dependency**: Full semantic vector search requires an active Hindsight instance (Cloud or local server). Offline environments rely on SQLite relational queries.
- **Single-Tenant Database**: The default database and memory bank (`HINDSIGHT_BANK_ID="incidentiq"`) are structured for single-team deployments. Multi-tenancy with organization-level bank isolation is planned for future releases.
- **Manual Triggering**: Investigations are currently initiated manually through the UI or REST API. Automated webhook integrations for PagerDuty, Datadog, and Prometheus Alertmanager are currently under development.

---

## Conclusion

Integrating persistent memory with [Hindsight](https://hindsight.vectorize.io/) fundamentally changes how software teams handle production outages. By capturing verified learnings from resolved incidents and surfacing them during new outages, MemoryOps replaces repetitive debugging with continuous organizational learning.

To build persistent memory into your own software agents, explore the [Hindsight GitHub repository](https://github.com/vectorize-io/hindsight) and review the official [Hindsight documentation](https://hindsight.vectorize.io/).
