import uuid
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Incident
from app.schemas import (
    IncidentCreate,
    IncidentUpdate,
    IncidentResolve,
    IncidentResponse,
    MemoryRecallQuery,
    MemoryReflectQuery,
    IncidentAnalysisRequest,
    IncidentAnalysisResponse,
    IncidentInvestigationResponse,
)
from app.hindsight_service import hindsight_service
from app.ai_service import ai_incident_service

router = APIRouter(prefix="/api/v1/incidents", tags=["Incidents"])
legacy_router = APIRouter(prefix="/api/incidents", tags=["Incidents (Legacy Route)"])

def parse_memory_item(item: Any, target_incident: Optional[Incident] = None) -> Dict[str, Any]:
    """
    Helper to convert Hindsight memory result into structured dict with robust parsing,
    strict self-match identification, and dynamic relevance scoring.
    """
    if isinstance(item, dict):
        d = item
    elif hasattr(item, "model_dump"):
        d = item.model_dump()
    elif hasattr(item, "__dict__"):
        d = item.__dict__
    else:
        d = {"text": str(item)}

    text_content = d.get("text") or d.get("content") or str(d)

    # Extract metadata dictionary if present
    metadata = d.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = {}

    incident_id = str(d.get("incident_id") or d.get("document_id") or (metadata.get("incident_id") if isinstance(metadata, dict) else "") or "")
    service = str(d.get("service") or (metadata.get("service") if isinstance(metadata, dict) else "") or "")
    error = str(d.get("error") or "")
    symptoms = str(d.get("symptoms") or "")
    root_cause = str(d.get("root_cause") or "")
    resolution = str(d.get("resolution") or "")
    post_mortem = str(d.get("post_mortem") or "")
    outcome = str(d.get("outcome") or (metadata.get("outcome") if isinstance(metadata, dict) else "") or "Resolved")
    date_str = str(d.get("created_at") or d.get("date") or "")

    lines = text_content.split("\n")
    for line in lines:
        line_clean = line.strip()
        if line_clean.startswith("Incident ID:"):
            incident_id = line_clean.replace("Incident ID:", "").strip()
        elif line_clean.startswith("Service:"):
            service = line_clean.replace("Service:", "").strip()
        elif line_clean.startswith("Error:"):
            error = line_clean.replace("Error:", "").strip()
        elif line_clean.startswith("Symptoms:"):
            symptoms = line_clean.replace("Symptoms:", "").strip()
        elif line_clean.startswith("Root Cause:"):
            root_cause = line_clean.replace("Root Cause:", "").strip()
        elif line_clean.startswith("Resolution:"):
            resolution = line_clean.replace("Resolution:", "").strip()
        elif line_clean.startswith("Post-mortem:"):
            post_mortem = line_clean.replace("Post-mortem:", "").strip()
        elif line_clean.startswith("Outcome:"):
            outcome = line_clean.replace("Outcome:", "").strip()

    # Calculate dynamic relevance score and detailed match reasoning
    relevance_score = 0
    relevance_reasons = []
    differences_noted = []

    if target_incident:
        target_service = (target_incident.service or "").lower()
        target_error = (target_incident.error or "").lower()
        target_symptoms = (target_incident.symptoms or "").lower()

        # Check service similarity
        if service and service.lower() == target_service:
            relevance_score += 40
            relevance_reasons.append(f"Same service ({service})")
        elif service and target_service:
            differences_noted.append(f"Different service ({service} vs {target_incident.service})")

        # Check error signature similarity
        if error and target_error:
            err_a = error.lower()
            err_b = target_error
            if err_a == err_b:
                relevance_score += 40
                relevance_reasons.append("Identical error signature")
            elif ("tls" in err_a and "tls" in err_b) or ("database" in err_a and "database" in err_b) or ("oom" in err_a and "oom" in err_b):
                relevance_score += 25
                relevance_reasons.append("Matching error domain")
            elif "502" in err_a and "502" in err_b:
                relevance_score += 10
                relevance_reasons.append("Shared HTTP 502 status code")
                differences_noted.append("Different underlying root cause mechanism")

        # Check verified resolution presence
        if root_cause and resolution:
            relevance_score += 15
            relevance_reasons.append("Verified root cause & resolution")

        # Check symptom overlap
        if symptoms and target_symptoms:
            shared_words = set(re.findall(r'\b\w{4,}\b', symptoms.lower())).intersection(
                set(re.findall(r'\b\w{4,}\b', target_symptoms))
            )
            stop_words = {"high", "http", "error", "errors", "failed", "failing", "service", "with", "from", "during", "requests", "request"}
            sig_words = [w for w in shared_words if w not in stop_words]
            if sig_words:
                relevance_score += 5
                relevance_reasons.append(f"Symptom overlap ({', '.join(sig_words[:2])})")

    relevance_text = " | ".join(relevance_reasons) if relevance_reasons else f"Historical {service or 'system'} incident memory."
    if differences_noted and relevance_reasons:
        relevance_text += f" (Note: {'; '.join(differences_noted)})"

    return {
        "incident_id": incident_id,
        "service": service,
        "error": error,
        "symptoms": symptoms,
        "root_cause": root_cause,
        "resolution": resolution,
        "post_mortem": post_mortem,
        "outcome": outcome,
        "date": date_str,
        "relevance": relevance_text,
        "relevance_score": relevance_score,
        "raw_text": text_content,
    }


async def perform_investigation_workflow(incident: Incident, db: Session) -> IncidentInvestigationResponse:
    # 1. Hindsight RECALL (Async)
    recall_query = f"Service: {incident.service} | Error: {incident.error} | Symptoms: {incident.symptoms}"
    recalled = await hindsight_service.arecall_memories(query=recall_query, max_tokens=2048)

    similar_incidents: List[Dict[str, Any]] = []
    previous_root_causes: List[str] = []
    previous_resolutions: List[str] = []

    memory_status = "unavailable"
    recall_status = "failed"
    recall_source = "hindsight_error"
    message = "Historical memory unavailable during investigation."

    if recalled.get("success") and recalled.get("results") is not None:
        raw_res = recalled["results"]
        items = []
        if isinstance(raw_res, dict):
            items = raw_res.get("results", []) or raw_res.get("memories", [])
        elif isinstance(raw_res, list):
            items = raw_res
        elif hasattr(raw_res, "results"):
            items = getattr(raw_res, "results") or []

        parsed_items = []
        target_id_upper = (incident.id or "").upper()

        for item in items:
            parsed = parse_memory_item(item, target_incident=incident)

            # STRICT SELF-MATCH EXCLUSION: Check incident_id, text content, and metadata for target incident ID
            m_id = parsed["incident_id"].upper() if parsed.get("incident_id") else ""
            raw_text = (parsed.get("raw_text") or "").upper()

            if target_id_upper and (m_id == target_id_upper or f"INCIDENT ID: {target_id_upper}" in raw_text or f"INCIDENT ID: {target_id_upper}\n" in raw_text):
                continue

            parsed_items.append(parsed)

        # Deduplicate recalled memories using stable incident_id and root_cause/resolution
        seen_keys = set()
        deduped_items = []
        for p in parsed_items:
            p_inc_id = (p.get("incident_id") or "").upper()
            p_rc = (p.get("root_cause") or "").strip()
            p_res = (p.get("resolution") or "").strip()
            p_svc = (p.get("service") or "").lower()
            p_err = (p.get("error") or "").lower()

            key = (p_inc_id, p_rc, p_res) if p_inc_id else (p_svc, p_err, p_rc)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_items.append(p)

        # Filter out memories with negligible relevance score (< 15) unless service matches
        target_svc_lower = (incident.service or "").lower()
        meaningful_items = [
            p for p in deduped_items
            if p["relevance_score"] >= 15 or (p.get("service") and p["service"].lower() == target_svc_lower)
        ]

        # Rank memories by relevance score descending
        ranked_items = sorted(meaningful_items, key=lambda x: x["relevance_score"], reverse=True)

        if len(ranked_items) > 0:
            memory_status = "ok"
            recall_status = "success"
            recall_source = "hindsight"
            for p in ranked_items[:5]:
                similar_incidents.append(p)
                # Extract structured evidence from verified historical records
                if p["root_cause"] and p["root_cause"] not in previous_root_causes:
                    previous_root_causes.append(p["root_cause"])
                if p["resolution"] and p["resolution"] not in previous_resolutions:
                    previous_resolutions.append(p["resolution"])
            message = f"Recalled {len(similar_incidents)} unique historical memories from Hindsight."
        else:
            memory_status = "empty"
            recall_status = "empty"
            recall_source = "hindsight"
            message = "Hindsight searched previous incidents but found no relevant historical experience for this query."
    else:
        memory_status = "unavailable"
        recall_status = "failed"
        recall_source = "hindsight_error"
        err_msg = recalled.get("error") or "Hindsight service unavailable"
        err_detail = recalled.get("detail") or ""
        message = f"Hindsight Recall Status: {err_msg}. {err_detail}".strip()

    # Create a filtered recalled_memories structure to pass to Groq AI
    filtered_recalled = dict(recalled)
    if memory_status == "ok":
        filtered_recalled["filtered_memories"] = similar_incidents
        filtered_recalled["previous_root_causes"] = previous_root_causes
        filtered_recalled["previous_resolutions"] = previous_resolutions

    # 2. Groq Analysis
    ai_res = await ai_incident_service.aanalyze_incident(
        service=incident.service,
        error=incident.error,
        symptoms=incident.symptoms,
        severity=incident.severity,
        recalled_memories=filtered_recalled,
    )

    analysis_status = "success" if ai_res.get("success") else "fallback"
    recommended_action = ai_res.get("recommended_action", "Investigate service logs and system metrics.")
    explanation = ai_res.get("reasoning", "Analysis based on current symptoms and historical incident recall.")

    # Persist AI recommendation to SQLite
    if recommended_action and incident.ai_recommendation != recommended_action:
        incident.ai_recommendation = recommended_action
        db.commit()
        db.refresh(incident)

    ai_analysis_summary = {
        "probable_root_cause": ai_res.get("probable_root_cause"),
        "confidence": ai_res.get("confidence"),
        "reasoning": ai_res.get("reasoning"),
        "supporting_historical_incidents": ai_res.get("supporting_historical_incidents", []) if memory_status == "ok" else [],
    }

    return IncidentInvestigationResponse(
        current_incident=IncidentResponse.model_validate(incident),
        similar_historical_incidents=similar_incidents,
        previous_root_causes=previous_root_causes,
        previous_resolutions=previous_resolutions,
        ai_analysis=ai_analysis_summary,
        recommended_action=recommended_action,
        explanation=explanation,
        analysis_status=analysis_status,
        memory_status=memory_status,
        message=message,
        recalled_memories_details=similar_incidents,
        recall_status=recall_status,
        recall_source=recall_source,
    )

@router.post("", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
@legacy_router.post("", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
async def create_incident(incident_in: IncidentCreate, db: Session = Depends(get_db)):
    incident_id = incident_in.id
    if incident_id:
        existing = db.query(Incident).filter(Incident.id == incident_id).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Incident with ID '{incident_id}' already exists."
            )
    else:
        incident_id = f"INC-{uuid.uuid4().hex[:6].upper()}"

    now = datetime.now(timezone.utc)
    is_resolved = incident_in.outcome.lower() == "resolved"
    has_details = bool(incident_in.root_cause and incident_in.resolution)

    incident = Incident(
        id=incident_id,
        service=incident_in.service,
        error=incident_in.error,
        symptoms=incident_in.symptoms,
        severity=incident_in.severity,
        root_cause=incident_in.root_cause,
        resolution=incident_in.resolution,
        post_mortem=incident_in.post_mortem,
        outcome=incident_in.outcome,
        created_at=now,
        resolved_at=now if is_resolved else None,
        memory_retained=False,
    )

    db.add(incident)
    db.commit()
    db.refresh(incident)

    # RETAIN only if created as resolved WITH resolution details and not previously retained
    if is_resolved and has_details:
        retain_res = await hindsight_service.aretain_incident(
            incident_id=incident.id,
            service=incident.service,
            error=incident.error,
            symptoms=incident.symptoms,
            severity=incident.severity,
            root_cause=incident.root_cause,
            resolution=incident.resolution,
            post_mortem=incident.post_mortem,
            outcome=incident.outcome,
        )
        if retain_res.get("success"):
            incident.memory_retained = True
            db.commit()
            db.refresh(incident)

    return incident

@router.get("", response_model=List[IncidentResponse])
@legacy_router.get("", response_model=List[IncidentResponse])
def get_incidents(
    service: Optional[str] = Query(None, description="Filter by service name"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    outcome: Optional[str] = Query(None, description="Filter by outcome/status"),
    db: Session = Depends(get_db),
):
    query = db.query(Incident)
    if service:
        query = query.filter(Incident.service.ilike(f"%{service}%"))
    if severity:
        query = query.filter(Incident.severity.ilike(severity))
    if outcome:
        query = query.filter(Incident.outcome.ilike(outcome))

    return query.order_by(Incident.created_at.desc()).all()

@router.post("/backfill-hindsight")
@legacy_router.post("/backfill-hindsight")
async def backfill_hindsight_memories(db: Session = Depends(get_db)):
    """
    Safely backfill all resolved incidents that have root causes and resolutions into Hindsight memory.
    Uses deterministic document_id deduplication so that existing memories are updated without creating duplicates.
    """
    resolved_incidents = db.query(Incident).filter(
        Incident.outcome.ilike("resolved"),
        Incident.resolution.isnot(None)
    ).all()

    total_count = len(resolved_incidents)
    successful_count = 0
    failed_count = 0
    details = []

    for inc in resolved_incidents:
        res = await hindsight_service.aretain_incident(
            incident_id=inc.id,
            service=inc.service,
            error=inc.error,
            symptoms=inc.symptoms,
            severity=inc.severity,
            root_cause=inc.root_cause,
            resolution=inc.resolution,
            post_mortem=inc.post_mortem,
            outcome=inc.outcome,
        )
        if res.get("success"):
            inc.memory_retained = True
            successful_count += 1
            details.append({"incident_id": inc.id, "status": "retained"})
        else:
            inc.memory_retained = False
            failed_count += 1
            details.append({"incident_id": inc.id, "status": "failed", "error": res.get("error")})

    db.commit()

    return {
        "success": failed_count == 0 and total_count > 0,
        "total_eligible_incidents": total_count,
        "retained_count": successful_count,
        "failed_count": failed_count,
        "details": details,
    }

@router.post("/recall")
@legacy_router.post("/recall")
async def recall_similar_incidents(query_in: MemoryRecallQuery, db: Session = Depends(get_db)):
    search_parts = []
    if query_in.query:
        search_parts.append(query_in.query)
    if query_in.service:
        search_parts.append(f"Service: {query_in.service}")
    if query_in.error:
        search_parts.append(f"Error: {query_in.error}")
    if query_in.symptoms:
        search_parts.append(f"Symptoms: {query_in.symptoms}")

    full_query = " | ".join(search_parts) if search_parts else "historical incidents"

    result = await hindsight_service.arecall_memories(
        query=full_query,
        max_tokens=2048,
        tags=query_in.tags,
    )

    memories_parsed = []
    memory_status = "unavailable"
    if result.get("success") and result.get("results") is not None:
        raw_res = result["results"]
        items = []
        if isinstance(raw_res, dict):
            items = raw_res.get("results", []) or raw_res.get("memories", [])
        elif isinstance(raw_res, list):
            items = raw_res

        if len(items) > 0:
            memory_status = "ok"
            for item in items[:5]:
                memories_parsed.append(parse_memory_item(item))
        else:
            memory_status = "empty"

    return {
        "success": result.get("success", False),
        "memory_status": memory_status,
        "query": full_query,
        "memories": memories_parsed,
        "results": result.get("results"),
        "raw_response": result,
    }

@router.post("/reflect")
@legacy_router.post("/reflect")
async def reflect_incident_patterns(query_in: MemoryReflectQuery, db: Session = Depends(get_db)):
    result = await hindsight_service.areflect_patterns(
        query=query_in.query,
        context=query_in.context,
    )
    return result

@router.post("/analyze", response_model=IncidentAnalysisResponse)
@legacy_router.post("/analyze", response_model=IncidentAnalysisResponse)
async def analyze_new_incident(analysis_in: IncidentAnalysisRequest):
    return await ai_incident_service.aanalyze_incident(
        service=analysis_in.service,
        error=analysis_in.error,
        symptoms=analysis_in.symptoms,
        severity=analysis_in.severity,
        custom_query=analysis_in.custom_query,
    )

@router.post("/{incident_id}/analyze", response_model=IncidentInvestigationResponse)
@legacy_router.post("/{incident_id}/analyze", response_model=IncidentInvestigationResponse)
async def analyze_existing_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found."
        )

    return await perform_investigation_workflow(incident, db)

@router.post("/{incident_id}/retain")
@legacy_router.post("/{incident_id}/retain")
async def retain_incident_memory(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found."
        )

    if incident.outcome.lower() != "resolved":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only resolved incidents with verified root cause and resolution can be retained into Hindsight."
        )

    if not (incident.root_cause or incident.resolution):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incident is missing root cause or resolution information required for memory retention."
        )

    retain_res = await hindsight_service.aretain_incident(
        incident_id=incident.id,
        service=incident.service,
        error=incident.error,
        symptoms=incident.symptoms,
        severity=incident.severity,
        root_cause=incident.root_cause,
        resolution=incident.resolution,
        post_mortem=incident.post_mortem,
        outcome=incident.outcome,
    )

    if retain_res.get("success"):
        incident.memory_retained = True
        db.commit()
        db.refresh(incident)
        return {
            "success": True,
            "message": f"Successfully retained memory for incident {incident_id} in Hindsight.",
            "incident": IncidentResponse.model_validate(incident),
        }
    else:
        incident.memory_retained = False
        db.commit()
        db.refresh(incident)
        return {
            "success": False,
            "message": f"Failed to retain memory in Hindsight: {retain_res.get('error')}",
            "detail": retain_res.get('detail'),
            "incident": IncidentResponse.model_validate(incident),
        }

@router.get("/{incident_id}", response_model=IncidentResponse)
@legacy_router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found."
        )
    return incident

@router.patch("/{incident_id}", response_model=IncidentResponse)
@legacy_router.patch("/{incident_id}", response_model=IncidentResponse)
async def update_incident(
    incident_id: str,
    incident_in: IncidentUpdate,
    db: Session = Depends(get_db),
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found."
        )

    update_data = incident_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(incident, field, value)

    if "outcome" in update_data and update_data["outcome"]:
        if update_data["outcome"].lower() == "resolved" and not incident.resolved_at:
            incident.resolved_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(incident)

    # RETAIN only if resolved, has resolution details, and NOT already retained
    if incident.outcome.lower() == "resolved" and incident.resolution and not incident.memory_retained:
        retain_res = await hindsight_service.aretain_incident(
            incident_id=incident.id,
            service=incident.service,
            error=incident.error,
            symptoms=incident.symptoms,
            severity=incident.severity,
            root_cause=incident.root_cause,
            resolution=incident.resolution,
            post_mortem=incident.post_mortem,
            outcome=incident.outcome,
        )
        if retain_res.get("success"):
            incident.memory_retained = True
            db.commit()
            db.refresh(incident)

    return incident

@router.post("/{incident_id}/resolve", response_model=IncidentResponse)
@legacy_router.post("/{incident_id}/resolve", response_model=IncidentResponse)
async def resolve_incident(
    incident_id: str,
    resolve_in: IncidentResolve,
    db: Session = Depends(get_db),
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found."
        )

    if resolve_in.root_cause:
        incident.root_cause = resolve_in.root_cause
    if resolve_in.post_mortem:
        incident.post_mortem = resolve_in.post_mortem
    incident.resolution = resolve_in.resolution
    incident.outcome = resolve_in.outcome
    incident.resolved_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(incident)

    # RETAIN only if NOT already retained
    if not incident.memory_retained:
        retain_res = await hindsight_service.aretain_incident(
            incident_id=incident.id,
            service=incident.service,
            error=incident.error,
            symptoms=incident.symptoms,
            severity=incident.severity,
            root_cause=incident.root_cause,
            resolution=incident.resolution,
            post_mortem=incident.post_mortem,
            outcome=incident.outcome,
        )
        if retain_res.get("success"):
            incident.memory_retained = True
            db.commit()
            db.refresh(incident)

    return incident
