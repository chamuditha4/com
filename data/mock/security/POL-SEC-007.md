---
doc_id: POL-SEC-007
title: Certificate Lifecycle Management Standard
department: security
document_type: policy
access_level: internal
created_date: '2026-04-02'
tags:
- certificates
- tls
- security
---

# Certificate Lifecycle Management Standard

## Background

This standard was introduced after repeated payment outages caused by expired certificates
(INC-PAY-2025-041 and INC-PAY-2026-011).

## Requirements

- Every TLS, mutual-TLS client and code-signing certificate must be registered in the central
certificate manager with a named owner.
- Certificates must be renewed automatically at least 30 days before expiry where technically possible.
- Expiry alerts are required at 30, 14 and 7 days and escalate to the owning team's on-call.
- Automated discovery scans run weekly to find certificates missing from the inventory.

## Exceptions

Certificates that cannot be renewed automatically, such as hardware security module certificates,
require a documented manual renewal plan approved by Platform Security.
