"""HTTP API tests: auth, RBAC, rate limiting, validation, SSE streaming, HITL and session isolation."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.container import build_container
from app.main import create_app
from mcp_server.server import server as mcp_server

PASSWORDS = {"viewer": "viewer-demo-pass", "analyst": "analyst-demo-pass", "admin": "admin-demo-pass"}


def _client(settings) -> TestClient:
    async def factory(s, stack):
        return await build_container(s, stack, mcp_target=mcp_server)

    return TestClient(create_app(settings, container_factory=factory), raise_server_exceptions=False)


@pytest.fixture
def client(test_settings):
    with _client(test_settings) as c:
        yield c


def login(client: TestClient, username: str) -> dict[str, str]:
    response = client.post("/auth/token", json={"username": username, "password": PASSWORDS[username]})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        event_line, data_line = block.split("\n", 1)
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


# --- auth & errors ------------------------------------------------------------------------------


def test_login_returns_token_with_role_and_clearance(client):
    response = client.post("/auth/token", json={"username": "analyst", "password": PASSWORDS["analyst"]})
    body = response.json()
    assert response.status_code == 200
    assert body["user"]["role"] == "analyst"
    assert body["user"]["clearance"] == ["public", "internal", "confidential"]
    assert "x-trace-id" in response.headers


def test_failed_login_is_structured_and_does_not_leak_which_field_was_wrong(client):
    response = client.post("/auth/token", json={"username": "analyst", "password": "wrong"})
    error = response.json()["error"]
    assert response.status_code == 401
    assert error == {
        "code": "unauthenticated",
        "message": "Invalid username or password.",
        "trace_id": response.headers["x-trace-id"],
    }


@pytest.mark.parametrize("header", [None, "Bearer not-a-jwt", "Basic dXNlcjpwYXNz"])
def test_protected_routes_reject_missing_or_invalid_tokens(client, header):
    headers = {"Authorization": header} if header else {}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


def test_unknown_route_returns_structured_404(client):
    assert client.get("/nope").json()["error"]["code"] == "not_found"


def test_health_endpoints(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["vector_store"] is True


# --- RBAC -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("username", "status"), [("viewer", 403), ("analyst", 403), ("admin", 200)])
def test_admin_endpoints_are_enforced_server_side(client, username, status):
    response = client.get("/admin/system", headers=login(client, username))
    assert response.status_code == status
    if status == 403:
        assert response.json()["error"]["details"] == {"required_permission": "admin"}
    else:
        config = response.json()["config"]
        assert not any("secret" in k or "key" in k for k in config)
        assert response.json()["catalog"]["total_documents"] == 31


# --- chat -------------------------------------------------------------------------------------------


def test_chat_returns_validated_cited_answer(client):
    response = client.post(
        "/chat",
        json={"message": "How do I rotate a TLS certificate?", "session_id": "s1"},
        headers=login(client, "viewer"),
    )
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "completed"
    assert body["answer"]["validation_passed"] and body["answer"]["citations"]
    assert body["trace"]["trace_id"] == response.headers["x-trace-id"]


def test_stream_emits_activity_then_final_answer(client):
    response = client.post(
        "/chat/stream",
        json={
            "message": "Summarize all outage reports related to payment failures during the last year and identify recurring root causes.",
            "session_id": "rlm",
        },
        headers=login(client, "analyst"),
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]

    assert names[0] == "start" and names[-1] == "done" and "final" in names
    trace_id = events[0][1]["trace_id"]
    activity = [data for name, data in events if name == "activity"]
    assert {"plan", "recursion", "aggregate", "validation"} <= {a["kind"] for a in activity}
    assert all(a["trace_id"] == trace_id for a in activity)
    final = next(data for name, data in events if name == "final")
    assert len(final["research_report"]["incidents"]) == 10


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "", "session_id": "s1"},
        {"message": "x" * 4001, "session_id": "s1"},
        {"message": "​​", "session_id": "s1"},
        {"message": "hello", "session_id": "../other-user"},
    ],
)
def test_invalid_chat_requests_are_rejected_without_echoing_input(client, payload):
    response = client.post("/chat", json=payload, headers=login(client, "viewer"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert "x" * 50 not in response.text


def test_per_user_rate_limit_returns_429_with_retry_after(test_settings):
    settings = test_settings.model_copy(update={"rate_limit_capacity": 2, "rate_limit_refill_per_second": 0.01})
    with _client(settings) as client:
        viewer, analyst = login(client, "viewer"), login(client, "analyst")
        codes = [
            client.post("/chat", json={"message": "hello", "session_id": "rl"}, headers=viewer).status_code
            for _ in range(3)
        ]
        assert codes == [200, 200, 429]

        limited = client.post("/chat", json={"message": "hello", "session_id": "rl"}, headers=viewer)
        assert int(limited.headers["retry-after"]) >= 1
        assert limited.json()["error"]["code"] == "rate_limited"
        # Buckets are per user: another user is unaffected.
        assert client.post("/chat", json={"message": "hello", "session_id": "rl"}, headers=analyst).status_code == 200


def test_sessions_are_isolated_per_user(client):
    client.post(
        "/chat",
        json={"message": "What is the password policy?", "session_id": "shared"},
        headers=login(client, "analyst"),
    )

    own = client.get("/chat/sessions/shared/history", headers=login(client, "analyst")).json()
    other = client.get("/chat/sessions/shared/history", headers=login(client, "viewer")).json()
    assert len(own["messages"]) == 2
    assert other["messages"] == []


def test_human_in_the_loop_approval_over_http(client):
    admin = login(client, "admin")
    paused = client.post(
        "/chat", json={"message": "Please reindex the knowledge base", "session_id": "hitl"}, headers=admin
    ).json()
    assert paused["status"] == "awaiting_approval"
    assert paused["approval"]["calls"][0]["tool"] == "admin_reindex_knowledge_base"

    blocked = client.post("/chat", json={"message": "hello", "session_id": "hitl"}, headers=admin)
    assert blocked.status_code == 409

    done = client.post("/chat/resume", json={"session_id": "hitl", "approved": True}, headers=admin).json()
    assert done["status"] == "completed"
    assert done["tool_calls"][0]["status"] == "succeeded"

    assert client.post("/chat/resume", json={"session_id": "hitl", "approved": True}, headers=admin).status_code == 409
