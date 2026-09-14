---
doc_id: RB-PAY-003
title: Payment Gateway Failover Runbook
department: payments
document_type: runbook
access_level: internal
created_date: '2025-09-01'
tags:
- runbook
- failover
- payments
---

# Payment Gateway Failover Runbook

## When to Use

Use this runbook when the primary payment gateway cluster in the Colombo data centre is unhealthy:
authorization success rate below 90% for more than 5 minutes, or health checks failing on more than
half of gateway nodes.

## Steps

1. Confirm the alert in the payments dashboard and declare an incident per POL-TEC-010.
2. Check the processor links (CardNet, LankaAcquire) to rule out a third-party outage.
3. Shift traffic to the secondary data centre with the traffic-manager command `failover payments-gw --to dc2`.
4. Verify authorization success rate recovers above 97% within 10 minutes.
5. Enable stand-in processing only if both data centres are affected.

## Rollback

Return traffic to the primary data centre only after 30 minutes of stable health checks.
