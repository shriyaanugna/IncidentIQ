# Building an Agent That Learns from Every Interaction with Hindsight

Every SRE and DevOps engineer knows the feeling of waking up to a 3 AM PagerDuty alert. Your primary gateway is returning HTTP 504 Gateway Timeouts, downstream microservices are failing readiness probes, and latency graphs are spiking vertically. You dive into terminal logs, search Slack channels, and search through unindexed post-mortem notes looking for clues. Deep down, you know someone on your team solved a nearly identical issue six months ago—perhaps a subtle connection pool session leak or an unindexed database query—but that knowledge is buried in closed tickets or isolated inside the head of an off-shift engineer.

This phenomenon is **organizational memory loss**, and it costs engineering teams hundreds of hours during critical production outages. When I set out to build **MemoryOps**, an AI-powered incident response workspace, my goal was clear: create an intelligent SRE copilot that learns from every resolved production incident and recalls those exact learnings during future outages.

However, off-the-shelf Large Language Models (LLMs) present a major obstacle: they lack persistent organizational context. When asked to diagnose an incident, generic LLMs tend to generate standard, generic, or downright dangerous advice, hallucinating remediation commands that have no bearing on your actual infrastructure.

To solve this, I built MemoryOps around [Hindsight](https://github.com/vectorize-io/hindsight), an open-source persistent memory engine designed for autonomous agents. By integrating Hindsight's vector memory bank with Groq's high-speed inference engine, I created a stateful agent that recalls past incident resolutions and uses them to ground its real-time recommendations.

In this article, I will walk you through the system architecture, dissect the exact code workflows for memory retention and recall, share a real-world incident use case, and highlight key engineering lessons learned along the way.

---

## Architecture Overview: Decoupling Memory from Reasoning

In traditional LLM applications, developers often try to pass massive unstructured logs directly into a prompt window. This approach is expensive, slow, and noisy. MemoryOps takes a different approach by decoupling the **long-term memory layer** from the **AI reasoning engine**.

Here is how the components interact in MemoryOps:

1. **Frontend Interface (React + Vite + Tailwind CSS)**: SREs manage active incidents, review AI analysis, trigger investigations, and explore historical memory banks via a dedicated dashboard.
2. **Backend API Gateway (FastAPI + SQLAlchemy)**: Manages incident lifecycle state in SQLite (`data/incidentiq.db`) and orchestrates the investigation workflow.
3. **Memory Engine (Hindsight Client)**: Implements persistent [Vectorize agent memory](https://vectorize.io/what-is-agent-memory) using semantic vector search across isolated memory banks.
4. **Reasoning Engine (Groq Cloud LLM)**: Synthesizes current incident symptoms with recalled historical memories to recommend concrete, evidence-backed remediation steps.

When an engineer reports a new incident, MemoryOps queries Hindsight using semantic recall. It pulls relevant past root causes and resolutions, formats them as grounded evidence, and injects them directly into Groq AI's prompt. Once the human engineer approves and resolves the incident, the final root cause and fix are retained back into Hindsight for future learning.

---

## Code Deep-Dive: Storage and Retrieval Workflows

Let's look at how MemoryOps implements these workflows using the official `hindsight-client` Python SDK.

### 1. Persistent Memory Storage (RETAIN)

When an incident is resolved, MemoryOps formats a structured textual experience document containing the service name, error signatures, symptoms, root cause, and successful resolution. This experience is retained into the Hindsight memory bank via `aretain_incident` in `backend/app/hindsight_service.py`:

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

Tagging the document with `service`, `severity`, and `outcome` enables metadata filtering alongside semantic vector search.

### 2. Semantic Memory Retrieval (RECALL)

During an active outage investigation, MemoryOps formulates a semantic query representing the new incident and calls `arecall_memories`:

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

Hindsight evaluates the semantic meaning of the query against the memory bank, returning relevant historical incidents even if the wording differs from past tickets.

### 3. Grounded AI Reasoning and Anti-Hallucination Controls

Once historical memories are recalled, they are passed into `AIIncidentService.aanalyze_incident` in `backend/app/ai_service.py`. To prevent LLM hallucinations, the system prompt strictly separates historical evidence from AI analysis and enforces a JSON response schema:

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

## Real-World Use Case: Resolving Database Connection Timeouts

To see MemoryOps and Hindsight in action, consider a scenario supported by the codebase seed data (`INC-101` and `INC-106` in `backend/app/seed.py`).

1. **Past Incident (`INC-101`)**:
   - **Service**: Payment API
   - **Error**: Database connection timeout
   - **Root Cause**: Connection pool exhaustion due to leaked unclosed DB sessions during a traffic spike.
   - **Resolution**: Increased connection pool size from 20 to 100 and deployed hotfix for session leak.
   - **Outcome**: Resolved and retained in Hindsight.

2. **New Incident (`INC-106`)**:
   - Six weeks later, an SRE receives an alert on Customer Relational Database: `"FATAL: remaining connection slots are reserved for non-replication superuser connections"`.
   - The engineer triggers an investigation in MemoryOps (`POST /api/v1/incidents/INC-106/analyze`).

3. **Hindsight Recall & Groq Reasoning**:
   - MemoryOps constructs the search query: `Service: Customer Relational Database | Error: Database connection problems | Symptoms: FATAL connection slots reserved`.
   - Hindsight executes vector search and recalls `INC-101` due to semantic similarity around database connection pool exhaustion.
   - Groq AI receives `INC-101` as grounded evidence and recommends inspecting active worker connections and configuring idle timeouts in PgBouncer.
   - The UI displays the recommendation alongside explicit evidence attribution: `"Supporting Historical Evidence: INC-101 - Database connection pool exhaustion"`.

4. **Resolution and Learning**:
   - The engineer configures PgBouncer idle timeouts, confirms recovery, and clicks **"Resolve & Retain"**.
   - MemoryOps updates `INC-106` in SQLite and calls `aretain_incident()`, adding another verified experience to the Hindsight bank.

---

## Technical Lessons Learned

Building MemoryOps yielded several key engineering insights into agentic memory design:

### 1. Async Event Loop Safety in FastAPI
When calling async SDK methods inside FastAPI route handlers, wrap sync wrappers carefully. Directly executing `asyncio.run()` inside an active event loop triggers runtime errors. Standardizing on native async methods (`aretain`, `arecall`, `areflect`) and using `await` throughout `backend/app/routers/incidents.py` keeps I/O non-blocking under load.

### 2. Graceful Fallbacks for External Memory Availability
External API services can experience credit exhaustion or temporary network outages. In `hindsight_service.py`, I implemented `is_insufficient_credits_error()` to detect HTTP 402 exceptions. When Hindsight Cloud returns an insufficient credits error, MemoryOps catches the exception and falls back to querying previously resolved incidents directly from SQLite (`db.query(Incident).filter(Incident.outcome.ilike("resolved"))`). This ensures SRE workflows remain functional even when vector memory is offline.

### 3. Guarding Against Duplicate Memory Retention
Re-retaining an incident every time metadata is edited pollutes vector memory banks with duplicate documents. To prevent this, the `Incident` database model includes a boolean `memory_retained` flag. RETAIN is executed only once when an incident transitions to `"Resolved"` with root cause details, setting `memory_retained = True` to block redundant calls.

### 4. Grounding Prompts to Eliminate LLM Hallucinations
Generative models excel at synthesis but struggle with strict truthfulness unless explicitly constrained. Forcing JSON schema output and requiring an explicit `supporting_historical_incidents` array transformed the LLM from a speculative assistant into a reliable, evidence-backed diagnostic engine.

---

## Current Limitations and Future Work

While MemoryOps demonstrates the power of persistent memory for SRE teams, it currently has a few limitations:
- **Vector Search Dependency**: Full semantic search depends on an active Hindsight instance (Cloud or local container). When offline, fallback relies on relational database matching.
- **Single-Tenant Storage**: The current implementation uses a single database and bank ID (`HINDSIGHT_BANK_ID="incidentiq"`). Multi-tenant support with role-based access control (RBAC) remains a future addition.
- **Manual Triggering**: Investigations are currently initiated via UI or REST API calls. Future improvements include automated webhooks for PagerDuty, Datadog, and Prometheus Alertmanager.

---

## Conclusion

Integrating persistent agent memory with [Hindsight](https://hindsight.vectorize.io/) transforms how software systems handle production outages. Instead of treating every incident as an isolated event, MemoryOps creates a self-learning platform where every resolved outage directly improves future response times.

If you are building AI agents for complex technical domains, explore the [Hindsight GitHub repository](https://github.com/vectorize-io/hindsight) and consult the official [Hindsight documentation](https://hindsight.vectorize.io/) to start building stateful, memory-augmented AI workflows today.
