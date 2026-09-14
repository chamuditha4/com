---
doc_id: RB-TEC-008
title: TLS Certificate Rotation Runbook
department: technology
document_type: runbook
access_level: internal
created_date: '2026-03-15'
tags:
- runbook
- certificates
---

# TLS Certificate Rotation Runbook

## Pre-checks

Confirm the certificate is registered in the central certificate manager and identify all services
consuming it with the dependency map.

## Rotation Steps

1. Request a new certificate from the internal CA with the same subject alternative names.
2. Deploy the certificate to the secrets vault; never commit certificate private keys to source control.
3. Trigger a rolling reload of consuming services.
4. Verify the TLS handshake with `openssl s_client` from a client host.
5. Update the expiry date in the certificate manager.

## Emergency Rotation

If a certificate has already expired and payments are failing, use the emergency CA issuance path and
page Platform Security. Target restoration is under 30 minutes.
