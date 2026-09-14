"""Commercial Bank operations MCP server (dummy data).

Exposes read-only operational tools over MCP Streamable HTTP. It runs as its own container,
reachable only on the internal Docker network. Authorization happens in the API before a call
is made (MCP tools require the `mcp` permission); see docs/SECURITY.md for the trust boundary.

Run:  python -m mcp_server.server   (from backend/)
"""

from __future__ import annotations

import os
from typing import Literal

from mcp.server.mcpserver import MCPServer

from mcp_server.data import ONCALL, PAYMENT_METRICS, SERVICE_STATUS, TICKETS

server = MCPServer(
    name="commercial-bank-ops",
    instructions="Read-only operational data for Commercial Bank payment services.",
)


@server.tool(description="Search operational incident tickets by service, status and opening date.")
def search_incident_tickets(
    service: str | None = None,
    status: Literal["open", "closed", "any"] = "any",
    opened_after: str | None = None,
    limit: int = 20,
) -> dict:
    """Return tickets filtered by service slug (e.g. 'card-authorization'), status and ISO date."""
    limit = max(1, min(limit, 50))
    rows = [
        t
        for t in TICKETS
        if (service is None or t["service"] == service)
        and (status == "any" or t["status"] == status)
        and (opened_after is None or t["opened"] >= opened_after)
    ]
    return {"count": len(rows[:limit]), "tickets": rows[:limit]}


@server.tool(description="Get the current operational status of a payment service, or all services.")
def get_service_status(service: str | None = None) -> dict:
    if service is None:
        return {"services": SERVICE_STATUS}
    if service not in SERVICE_STATUS:
        return {"error": f"unknown service '{service}'", "known_services": sorted(SERVICE_STATUS)}
    return {"service": service, **SERVICE_STATUS[service]}


@server.tool(description="Get the current on-call roster for an engineering team.")
def get_oncall_roster(team: str) -> dict:
    if team not in ONCALL:
        return {"error": f"unknown team '{team}'", "known_teams": sorted(ONCALL)}
    return {"team": team, **ONCALL[team]}


@server.tool(description="Get monthly payment volume (thousands), failed transaction count and incident count.")
def get_payment_metrics(from_month: str | None = None, to_month: str | None = None) -> dict:
    """Months are 'YYYY-MM' strings, inclusive."""
    months = {
        m: v
        for m, v in PAYMENT_METRICS.items()
        if (from_month is None or m >= from_month) and (to_month is None or m <= to_month)
    }
    return {"unit_volume": "thousand transactions", "months": months}


def main() -> None:
    server.run(
        transport="streamable-http",
        host=os.getenv("MCP_HOST", "0.0.0.0"),  # noqa: S104 - container-internal
        port=int(os.getenv("MCP_PORT", "8765")),
        stateless_http=True,  # no server-side sessions: any replica can serve any request
    )


if __name__ == "__main__":
    main()
