import json
import logging
from typing import Any, Dict, List, Optional
from groq import Groq
from app.config import settings
from app.hindsight_service import hindsight_service

logger = logging.getLogger("incidentiq.ai_service")

SYSTEM_PROMPT = """You are MemoryOps AI, an expert SRE/DevOps incident response assistant.
Your task is to analyze an incoming IT/DevOps incident using recalled historical incident memories from Hindsight.

CRITICAL INSTRUCTIONS FOR AI GROUNDING & REASONING:
1. Distinguish clearly between verified historical facts (retrieved from Hindsight memories) and your own AI analysis/hypotheses.
2. NEVER invent or fabricate historical incidents or incident IDs. Only reference historical incidents that are explicitly present in the provided RECALLED HISTORICAL MEMORIES.
3. NEVER refer to the current incident being analyzed as a historical precedent.
4. Determine confidence strictly based on evidence quality:
   - "high": High relevance historical match exists with identical or near-identical service/error signatures and verified resolution.
   - "medium": Moderate historical match exists or partial error pattern overlap across related services.
   - "low": No relevant historical memories exist, or only weak generic symptom overlap exists across completely unrelated services.
5. If no relevant historical incidents exist or match, explicitly state that no relevant prior resolutions were found in Hindsight and base your analysis solely on current symptoms and general SRE best practices with "low" or "medium" confidence.
6. Return your output STRICTLY as a valid JSON object matching this exact schema:
{
  "probable_root_cause": "Detailed explanation of the probable root cause",
  "recommended_action": "Specific step-by-step remediation or investigation steps",
  "confidence": "high" | "medium" | "low",
  "reasoning": "Clear explanation distinguishing historical evidence from general model reasoning",
  "supporting_historical_incidents": [
    "INC-XXX: Summary of relevant historical incident"
  ]
}
"""

class AIIncidentService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not configured.")
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client

    async def aanalyze_incident(
        self,
        service: str,
        error: str,
        symptoms: str,
        severity: str = "medium",
        custom_query: Optional[str] = None,
        recalled_memories: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Async version of incident analysis.
        1. Recalls similar historical incidents from Hindsight asynchronously (if not passed).
        2. Sends current incident details and recalled memories to Groq LLM.
        3. Returns structured analysis with clear memory_status and analysis_status.
        """
        if recalled_memories is None:
            recall_query = custom_query or f"Service: {service} | Error: {error} | Symptoms: {symptoms}"
            try:
                recalled_memories = await hindsight_service.arecall_memories(query=recall_query)
                if not (recalled_memories and recalled_memories.get("success")):
                    sync_res = hindsight_service.recall_memories(query=recall_query)
                    if sync_res and sync_res.get("success"):
                        recalled_memories = sync_res
            except Exception:
                recalled_memories = hindsight_service.recall_memories(query=recall_query)

        memory_status = "unavailable"
        memories_text = "No historical memories retrieved because Hindsight memory was unavailable."

        filtered_items = []
        if recalled_memories and isinstance(recalled_memories, dict):
            filtered_items = recalled_memories.get("filtered_memories", [])

        if recalled_memories and recalled_memories.get("success") and recalled_memories.get("results") is not None:
            raw_res = recalled_memories["results"]
            items = filtered_items if filtered_items else []
            if not items and "filtered_memories" not in recalled_memories:
                if isinstance(raw_res, dict):
                    items = raw_res.get("results", []) or raw_res.get("memories", [])
                elif isinstance(raw_res, list):
                    items = raw_res
                elif hasattr(raw_res, "results"):
                    items = getattr(raw_res, "results") or []

            if len(items) > 0:
                memory_status = "ok"
                serializable_items = []
                for it in items:
                    if hasattr(it, "model_dump"):
                        serializable_items.append(it.model_dump())
                    elif isinstance(it, dict):
                        serializable_items.append(it)
                    elif hasattr(it, "__dict__"):
                        serializable_items.append(it.__dict__)
                    else:
                        serializable_items.append(str(it))
                memories_text = json.dumps(serializable_items, indent=2, default=str)
            else:
                memory_status = "empty"
                memories_text = "Hindsight search succeeded, but no unique relevant historical memories were found."

        try:
            user_prompt = f"""--- CURRENT INCIDENT DETAILS ---
Service: {service}
Error: {error}
Symptoms: {symptoms}
Severity: {severity}

--- RECALLED HISTORICAL MEMORIES (FROM HINDSIGHT) ---
{memories_text}

Analyze the current incident now and respond strictly with the JSON schema requested.
"""

            client = self.client
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            response_text = completion.choices[0].message.content
            parsed = json.loads(response_text)

            raw_supporting = parsed.get("supporting_historical_incidents", [])
            supporting = raw_supporting if isinstance(raw_supporting, list) else []

            # Bound confidence evaluation
            confidence = parsed.get("confidence", "medium")
            if memory_status in ["empty", "unavailable"]:
                confidence = "low"
            elif memory_status == "ok" and not filtered_items:
                confidence = "low"
            elif memory_status == "ok" and filtered_items:
                # Check top relevance score
                top_score = max([item.get("relevance_score", 0) for item in filtered_items], default=0)
                if top_score < 40 and confidence == "high":
                    confidence = "medium"

            return {
                "success": True,
                "analysis_status": "success",
                "memory_status": memory_status,
                "message": f"Incident analyzed with memory status: {memory_status}",
                "service": service,
                "error": error,
                "probable_root_cause": parsed.get("probable_root_cause", "Unknown root cause"),
                "recommended_action": parsed.get("recommended_action", "Investigate service logs"),
                "confidence": confidence,
                "reasoning": parsed.get("reasoning", "Analysis generated from incident details"),
                "supporting_historical_incidents": supporting if memory_status == "ok" else [],
                "recalled_memories_used": recalled_memories if memory_status != "unavailable" else None,
            }

        except json.JSONDecodeError as jde:
            logger.error(f"Failed to parse Groq AI JSON response: {jde}")
            return self._fallback_analysis(
                service, error, symptoms, severity, recalled_memories, memory_status,
                error_msg=f"Failed to parse AI response JSON: {jde}"
            )
        except ValueError as ve:
            logger.error(f"Groq API configuration error: {ve}")
            return self._fallback_analysis(
                service, error, symptoms, severity, recalled_memories, memory_status,
                error_msg=f"Groq API Key not configured: {ve}"
            )
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return self._fallback_analysis(
                service, error, symptoms, severity, recalled_memories, memory_status,
                error_msg=f"Groq AI service error: {e}"
            )

    def analyze_incident(
        self,
        service: str,
        error: str,
        symptoms: str,
        severity: str = "medium",
        custom_query: Optional[str] = None,
        recalled_memories: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sync wrapper for analyze_incident.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return self._fallback_analysis(
                service, error, symptoms, severity, recalled_memories, "unavailable",
                "Sync analyze called inside running loop"
            )
        else:
            return asyncio.run(self.aanalyze_incident(service, error, symptoms, severity, custom_query, recalled_memories))

    def _fallback_analysis(
        self,
        service: str,
        error: str,
        symptoms: str,
        severity: str,
        recalled_memories: Optional[Dict[str, Any]],
        memory_status: str,
        error_msg: str,
    ) -> Dict[str, Any]:
        """Graceful fallback when Groq API is unavailable or unconfigured."""
        supporting = []
        if recalled_memories and recalled_memories.get("success") and recalled_memories.get("results") and memory_status == "ok":
            supporting.append("Historical memories retrieved from Hindsight (Groq AI unavailable)")

        return {
            "success": False,
            "analysis_status": "fallback",
            "memory_status": memory_status,
            "message": f"Incident analyzed using rule-based heuristic. ({error_msg})",
            "error_detail": error_msg,
            "service": service,
            "error": error,
            "probable_root_cause": f"Potential issue in service '{service}' related to error: {error}",
            "recommended_action": f"Check logs and metrics for '{service}'. Verify database/network connection and service health.",
            "confidence": "low",
            "reasoning": f"Fallback rule-based heuristic applied because AI analysis was unavailable ({error_msg}).",
            "supporting_historical_incidents": supporting if memory_status == "ok" else [],
            "recalled_memories_used": recalled_memories,
        }

ai_incident_service = AIIncidentService()
