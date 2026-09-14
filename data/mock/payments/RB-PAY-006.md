---
doc_id: RB-PAY-006
title: Ledger Database Connection Pool Tuning Guide
department: payments
document_type: runbook
access_level: internal
created_date: '2026-07-10'
tags:
- runbook
- database
- capacity
---

# Ledger Database Connection Pool Tuning Guide

## Background

Three incidents in nine months (INC-PAY-2025-044, INC-PAY-2026-019, INC-PAY-2026-033) were caused by
ledger connection pool exhaustion.

## Sizing Formula

Pool size per service instance = (peak concurrent requests × average query time) ÷ target latency, with
30% headroom. Batch workloads must use a separate pool sized from worker parallelism.

## Monitoring

Alert when pool utilisation exceeds 80% for 2 minutes and when connection acquisition time exceeds 200 ms.

## Peak Calendar

Salary day (last working day), month end and festive seasons (April and December) require pre-scaling.
