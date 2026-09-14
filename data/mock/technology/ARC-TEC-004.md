---
doc_id: ARC-TEC-004
title: Core Banking Integration Architecture
department: technology
document_type: architecture
access_level: confidential
created_date: '2026-02-12'
tags:
- architecture
- core-banking
---

# Core Banking Integration Architecture

## Integration Pattern

Channels integrate with the core banking system through an event-driven integration layer. Account
postings are published to a durable event stream and consumed by the payments ledger.

## Security Zones

The core banking system sits in the highest security zone. Only the integration layer may call core
banking APIs, using mutual TLS and service identities.

## Modernization Roadmap

The 2026–2028 roadmap replaces batch file interfaces with real-time APIs and decommissions the legacy
message queue by Q4 2027.
