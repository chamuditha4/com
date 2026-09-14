"""Dummy enterprise operational data served by the MCP server.

Ticket ids and dates are consistent with the incident reports in `data/mock`, so an analyst can
join "what the documents say" with "what the ticketing system says" in one conversation.
"""

from __future__ import annotations

TICKETS = [
    {"ticket_id": "OPS-18231", "incident_ref": "INC-PAY-2025-041", "service": "card-authorization", "status": "closed", "priority": "P1", "opened": "2025-10-03", "closed": "2025-10-03", "summary": "Card declines - processor mTLS certificate expired", "assignee_team": "payments-engineering"},
    {"ticket_id": "OPS-18644", "incident_ref": "INC-PAY-2025-044", "service": "payments-ledger", "status": "closed", "priority": "P1", "opened": "2025-10-31", "closed": "2025-11-01", "summary": "Ledger connection pool exhausted on salary day", "assignee_team": "payments-engineering"},
    {"ticket_id": "OPS-19410", "incident_ref": "INC-PAY-2025-052", "service": "card-authorization", "status": "closed", "priority": "P2", "opened": "2025-12-12", "closed": "2025-12-13", "summary": "CardNet processor timeouts", "assignee_team": "vendor-management"},
    {"ticket_id": "OPS-19877", "incident_ref": "INC-PAY-2026-003", "service": "instant-payments", "status": "closed", "priority": "P2", "opened": "2026-01-09", "closed": "2026-01-10", "summary": "Instant payments rejected after release 4.18", "assignee_team": "payments-engineering"},
    {"ticket_id": "OPS-20512", "incident_ref": "INC-PAY-2026-011", "service": "mobile-wallet", "status": "closed", "priority": "P2", "opened": "2026-02-27", "closed": "2026-02-27", "summary": "Wallet top-ups failing - internal gateway cert expired", "assignee_team": "platform-security"},
    {"ticket_id": "OPS-21140", "incident_ref": "INC-PAY-2026-019", "service": "bill-payments", "status": "closed", "priority": "P2", "opened": "2026-03-31", "closed": "2026-04-01", "summary": "Month-end bill payment failures - pool saturation", "assignee_team": "data-platform"},
    {"ticket_id": "OPS-22035", "incident_ref": "INC-PAY-2026-027", "service": "cross-border-payments", "status": "closed", "priority": "P2", "opened": "2026-05-15", "closed": "2026-05-16", "summary": "FX routing flag enabled at 100%", "assignee_team": "devops"},
    {"ticket_id": "OPS-22790", "incident_ref": "INC-PAY-2026-033", "service": "standing-orders", "status": "closed", "priority": "P2", "opened": "2026-06-30", "closed": "2026-07-01", "summary": "Standing order batch failed - ledger pool exhaustion", "assignee_team": "payments-engineering"},
    {"ticket_id": "OPS-23311", "incident_ref": "INC-PAY-2026-038", "service": "merchant-qr", "status": "closed", "priority": "P3", "opened": "2026-07-24", "closed": "2026-07-24", "summary": "LankaAcquire outage - QR payments failing", "assignee_team": "payments-product"},
    {"ticket_id": "OPS-23958", "incident_ref": "INC-PAY-2026-045", "service": "swift-gateway", "status": "closed", "priority": "P1", "opened": "2026-08-21", "closed": "2026-08-22", "summary": "SWIFT signing blocked - HSM certificate expired", "assignee_team": "platform-security"},
    {"ticket_id": "OPS-24402", "incident_ref": None, "service": "payments-ledger", "status": "open", "priority": "P3", "opened": "2026-09-08", "closed": None, "summary": "Pool utilisation at 78% during salary-day rehearsal", "assignee_team": "payments-engineering"},
    {"ticket_id": "OPS-24417", "incident_ref": None, "service": "card-authorization", "status": "open", "priority": "P2", "opened": "2026-09-11", "closed": None, "summary": "CardNet client certificate expires in 21 days - renewal automation failed", "assignee_team": "platform-security"},
]

SERVICE_STATUS = {
    "card-authorization": {"status": "degraded", "success_rate_24h": 97.9, "note": "Elevated processor latency since 09:40"},
    "instant-payments": {"status": "operational", "success_rate_24h": 99.6, "note": ""},
    "payments-ledger": {"status": "operational", "success_rate_24h": 99.9, "note": "Pool utilisation peak 64%"},
    "bill-payments": {"status": "operational", "success_rate_24h": 99.4, "note": ""},
    "cross-border-payments": {"status": "operational", "success_rate_24h": 99.1, "note": ""},
    "swift-gateway": {"status": "operational", "success_rate_24h": 100.0, "note": ""},
    "mobile-wallet": {"status": "maintenance", "success_rate_24h": 98.8, "note": "Planned maintenance 23:00-01:00"},
    "merchant-qr": {"status": "operational", "success_rate_24h": 99.2, "note": ""},
}

ONCALL = {
    "payments-engineering": {"primary": "Nimal Perera", "secondary": "Ayesha Fernando", "escalation": "Head of Payments Engineering"},
    "platform-security": {"primary": "Kavindu Silva", "secondary": "Dilini Jayawardena", "escalation": "Platform Security Manager"},
    "sre": {"primary": "Ruwan Wickramasinghe", "secondary": "Tharushi de Alwis", "escalation": "SRE Lead"},
    "devops": {"primary": "Sahan Gunasekara", "secondary": "Ishara Bandara", "escalation": "DevOps Manager"},
}

# Monthly payment volumes (thousands of transactions) and failure counts, trailing 12 months.
PAYMENT_METRICS = {
    "2025-10": {"volume_k": 18450, "failed": 27700, "incidents": 2},
    "2025-11": {"volume_k": 17990, "failed": 4100, "incidents": 0},
    "2025-12": {"volume_k": 21340, "failed": 30200, "incidents": 1},
    "2026-01": {"volume_k": 18020, "failed": 11800, "incidents": 1},
    "2026-02": {"volume_k": 17110, "failed": 9200, "incidents": 1},
    "2026-03": {"volume_k": 19060, "failed": 16700, "incidents": 1},
    "2026-04": {"volume_k": 20480, "failed": 3900, "incidents": 0},
    "2026-05": {"volume_k": 18870, "failed": 5800, "incidents": 1},
    "2026-06": {"volume_k": 19320, "failed": 45100, "incidents": 1},
    "2026-07": {"volume_k": 19550, "failed": 12400, "incidents": 1},
    "2026-08": {"volume_k": 19930, "failed": 4600, "incidents": 1},
    "2026-09": {"volume_k": 8120, "failed": 1300, "incidents": 0},
}
