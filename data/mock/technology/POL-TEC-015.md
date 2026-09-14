---
doc_id: POL-TEC-015
title: Change Management Policy
department: technology
document_type: policy
access_level: internal
created_date: '2025-11-10'
tags:
- change-management
- deployment
---

# Change Management Policy

## Scope

All changes to production systems, including code deployments, infrastructure changes, database
changes, configuration changes and feature-flag changes.

## Change Types

- **Standard changes** are pre-approved and low risk, and must still use the automated pipeline.
- **Normal changes** require a change ticket, peer review and Change Advisory Board approval for
critical services.
- **Emergency changes** may be deployed to restore service and must be reviewed retrospectively
within 2 business days.

## Progressive Delivery

Changes to payment services must be released through canary deployment covering at least 5% of
traffic for 15 minutes with automated rollback on error-rate regression.
