import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.ai_service import AIIncidentService

client = TestClient(app)

def test_ai_service_analyze_incident_success():
    mock_groq_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "probable_root_cause": "Database connection pool exhaustion",
        "recommended_action": "Increase max_connections parameter from 20 to 100",
        "confidence": "high",
        "reasoning": "INC-101 historical incident showed identical Gateway Timeout symptoms and was resolved by expanding the pool.",
        "supporting_historical_incidents": ["INC-101: Payment API database connection timeout"]
    })
    mock_completion.choices = [mock_choice]
    mock_groq_client.chat.completions.create.return_value = mock_completion

    service = AIIncidentService(api_key="gsk_fake_key")
    service._client = mock_groq_client

    with patch("app.ai_service.hindsight_service.recall_memories") as mock_recall:
        recalled = {
            "success": True,
            "results": {"memories": [{"incident_id": "INC-101"}]},
            "filtered_memories": [{"incident_id": "INC-101", "relevance_score": 80}]
        }

        res = service.analyze_incident(
            service="Payment API",
            error="Connection timeout",
            symptoms="HTTP 504 Gateway Timeout",
            recalled_memories=recalled
        )

        assert res["success"] is True
        assert res["probable_root_cause"] == "Database connection pool exhaustion"
        assert res["confidence"] == "high"
        assert "INC-101" in res["supporting_historical_incidents"][0]

def test_ai_service_missing_api_key_fallback():
    service = AIIncidentService(api_key=None)

    with patch("app.ai_service.hindsight_service.recall_memories") as mock_recall:
        mock_recall.return_value = {"success": True, "results": []}

        res = service.analyze_incident(
            service="Payment API",
            error="Connection timeout",
            symptoms="504 Gateway Timeout"
        )

        assert res["success"] is False
        assert res["confidence"] == "low"
        assert "not configured" in res["error_detail"].lower()

def test_ai_service_json_parse_error_fallback():
    mock_groq_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "INVALID_JSON_RESPONSE"
    mock_completion.choices = [mock_choice]
    mock_groq_client.chat.completions.create.return_value = mock_completion

    service = AIIncidentService(api_key="gsk_fake_key")
    service._client = mock_groq_client

    with patch("app.ai_service.hindsight_service.recall_memories") as mock_recall:
        mock_recall.return_value = {"success": True, "results": []}

        res = service.analyze_incident(
            service="Payment API",
            error="Connection timeout",
            symptoms="504 Gateway Timeout"
        )

        assert res["success"] is False
        assert res["confidence"] == "low"
        assert "failed to parse" in res["error_detail"].lower()

def test_analyze_new_incident_api_endpoint():
    with patch("app.routers.incidents.ai_incident_service.aanalyze_incident") as mock_analyze:
        mock_analyze.return_value = {
            "success": True,
            "service": "Checkout Service",
            "error": "Payment gateway down",
            "probable_root_cause": "Third party payment provider gateway offline",
            "recommended_action": "Switch payment provider fallback flag to Braintree",
            "confidence": "high",
            "reasoning": "Retrieved past incident showed similar third party API outage.",
            "supporting_historical_incidents": ["INC-101: Payment API database connection timeout"]
        }

        response = client.post("/api/v1/incidents/analyze", json={
            "service": "Checkout Service",
            "error": "Payment gateway down",
            "symptoms": "Spike in 502 Bad Gateway responses during checkout"
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["probable_root_cause"] == "Third party payment provider gateway offline"

def test_analyze_existing_incident_api_endpoint():
    with patch("app.routers.incidents.hindsight_service.arecall_memories") as mock_recall, \
         patch("app.routers.incidents.ai_incident_service.aanalyze_incident") as mock_analyze:

        mock_recall.return_value = {"success": True, "results": []}
        mock_analyze.return_value = {
            "success": True,
            "service": "Payment API",
            "error": "Database connection timeout",
            "probable_root_cause": "Database connection pool exhaustion",
            "recommended_action": "Increase connection pool size",
            "confidence": "high",
            "reasoning": "Historical evidence match.",
            "supporting_historical_incidents": ["INC-101: Payment API database connection timeout"]
        }

        response = client.post("/api/v1/incidents/INC-101/analyze")
        assert response.status_code == 200
        data = response.json()
        assert data["current_incident"]["id"] == "INC-101"
        assert data["recommended_action"] == "Increase connection pool size"


# --- 10 NEW TEST CASES FOR GROQ MODEL CONFIGURATION, FALLBACKS, AND WORKFLOWS ---

def test_groq_model_loaded_from_settings():
    """1. Test that default GROQ_MODEL is correctly loaded from Settings."""
    from app.config import settings
    service = AIIncidentService(api_key="gsk_test")
    assert service.model == settings.GROQ_MODEL
    assert service.model == "llama-3.1-8b-instant"


def test_groq_custom_model_override():
    """2. Test that passing a custom model override to AIIncidentService is respected."""
    service = AIIncidentService(api_key="gsk_test", model="qwen/qwen3.8-27b")
    assert service.model == "qwen/qwen3.8-27b"


def test_groq_api_call_passes_configured_model():
    """3. Test that the Groq API completion request uses the configured model."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "probable_root_cause": "Test root cause",
        "recommended_action": "Test action",
        "confidence": "medium",
        "reasoning": "Test reasoning",
        "supporting_historical_incidents": []
    })
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    service = AIIncidentService(api_key="gsk_test", model="openai/gpt-oss-20b")
    service._client = mock_client

    res = service.analyze_incident(service="Auth", error="401 Unauthorized", symptoms="Login failed")

    assert res["success"] is True
    # Verify model kwarg passed to create()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-oss-20b"


def test_groq_model_not_found_triggers_fallback():
    """4. Test that a model_not_found error triggers the fallback heuristic with detail."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception(
        "Error code: 404 - The model 'obsolete-model' does not exist or you do not have access to it. code: model_not_found"
    )

    service = AIIncidentService(api_key="gsk_test", model="obsolete-model")
    service._client = mock_client

    res = service.analyze_incident(service="Auth", error="401", symptoms="Login failed")

    assert res["success"] is False
    assert res["confidence"] == "low"
    assert "model_not_found" in res["error_detail"] or "obsolete-model" in res["error_detail"]


def test_groq_rate_limit_error_triggers_fallback():
    """5. Test that a 429 rate limit error triggers fallback gracefully."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Error code: 429 - Rate limit reached for model")

    service = AIIncidentService(api_key="gsk_test")
    service._client = mock_client

    res = service.analyze_incident(service="Billing", error="500", symptoms="Checkout error")

    assert res["success"] is False
    assert res["confidence"] == "low"
    assert "Rate limit" in res["error_detail"]


def test_groq_network_connection_error_triggers_fallback():
    """6. Test that network connection exceptions trigger fallback gracefully."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("Failed to connect to api.groq.com")

    service = AIIncidentService(api_key="gsk_test")
    service._client = mock_client

    res = service.analyze_incident(service="Billing", error="500", symptoms="Timeout")

    assert res["success"] is False
    assert "ConnectionError" in res["error_detail"] or "Failed to connect" in res["error_detail"]


def test_groq_missing_json_fields_assigned_defaults():
    """7. Test that partial JSON responses safely receive default fallbacks for missing keys."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    # Response missing 'confidence' and 'supporting_historical_incidents'
    mock_choice.message.content = json.dumps({
        "probable_root_cause": "Memory leak in heap",
        "recommended_action": "Restart container"
    })
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    service = AIIncidentService(api_key="gsk_test")
    service._client = mock_client

    res = service.analyze_incident(service="Worker", error="OOM Killed", symptoms="High RAM")

    assert res["success"] is True
    assert res["probable_root_cause"] == "Memory leak in heap"
    assert res["recommended_action"] == "Restart container"
    assert res["confidence"] in ["low", "medium"]
    assert res["supporting_historical_incidents"] == []  # Default


def test_groq_async_custom_recall_query_passed_to_hindsight():
    """8. Test that custom_query parameter is forwarded to Hindsight recall in aanalyze_incident."""
    import pytest
    from unittest.mock import AsyncMock

    service = AIIncidentService(api_key="gsk_test")

    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "probable_root_cause": "Query test cause",
        "recommended_action": "Query test action",
        "confidence": "high",
        "reasoning": "Query reasoning",
        "supporting_historical_incidents": []
    })
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion
    service._client = mock_client

    with patch("app.ai_service.hindsight_service.arecall_memories", new_callable=AsyncMock) as mock_arecall:
        mock_arecall.return_value = {"success": True, "results": []}

        import asyncio
        res = asyncio.run(service.aanalyze_incident(
            service="Database",
            error="Lock timeout",
            symptoms="Slow queries",
            custom_query="Custom lock timeout recall query"
        ))

        assert res["success"] is True
        mock_arecall.assert_called_once_with(query="Custom lock timeout recall query")


def test_groq_grounded_historical_incidents_parsed():
    """9. Test that historical incident references in Groq prompt response are preserved."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "probable_root_cause": "PgBouncer pool exhaustion",
        "recommended_action": "Increase PgBouncer pool_size from 20 to 50",
        "confidence": "high",
        "reasoning": "INC-106 matches symptoms exactly.",
        "supporting_historical_incidents": [
            "INC-106: Database connection leak in auth-service"
        ]
    })
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    service = AIIncidentService(api_key="gsk_test")
    service._client = mock_client

    recalled = {
        "success": True,
        "results": {"memories": [{"incident_id": "INC-106"}]}
    }

    res = service.analyze_incident(
        service="Auth",
        error="FATAL: remaining connection slots reserved",
        symptoms="DB error",
        recalled_memories=recalled
    )

    assert res["success"] is True
    assert len(res["supporting_historical_incidents"]) == 1
    assert "INC-106" in res["supporting_historical_incidents"][0]


def test_fallback_preserves_recalled_memories_when_groq_fails():
    """10. Test that fallback analysis retains recalled historical memories when Groq fails."""
    service = AIIncidentService(api_key=None)  # Triggers fallback
    recalled = {
        "success": True,
        "results": [
            {"incident_id": "INC-101", "root_cause": "Pool exhausted", "resolution": "Expanded pool"}
        ]
    }

    res = service.analyze_incident(
        service="Payment API",
        error="Timeout",
        symptoms="504 Gateway Timeout",
        recalled_memories=recalled
    )

    assert res["success"] is False
    assert res["confidence"] == "low"
    assert res["recalled_memories_used"] == recalled
    assert len(res["supporting_historical_incidents"]) > 0
    assert "Historical memories retrieved from Hindsight" in res["supporting_historical_incidents"][0]
