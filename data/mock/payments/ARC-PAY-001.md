---
doc_id: ARC-PAY-001
title: Payments Platform Architecture Overview
department: payments
document_type: architecture
access_level: internal
created_date: '2025-09-20'
tags:
- architecture
- payments
---

# Payments Platform Architecture Overview

## Components

- **Payment API** — entry point for mobile, internet banking and partner channels.
- **Card Authorization Service** — connects to CardNet over mutual TLS.
- **Instant Payments Service** — connects to the national clearing switch for real-time transfers.
- **Payments Ledger** — PostgreSQL-backed double-entry ledger with read replicas.
- **SWIFT Gateway** — signs outbound messages with keys held in a hardware security module.

## Resilience

Services run active-active across two data centres. External processors are accessed through circuit
breakers. Payment instructions are persisted before acknowledgement so they can be replayed.

## Known Risks

Shared ledger connection pools and manually managed certificates were identified as architectural
risks in the 2026 reliability review.
