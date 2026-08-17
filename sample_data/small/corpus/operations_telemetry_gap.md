# Operations Runbook: Telemetry Gaps

Document ID: `OPS-TELEM-GAP`
Document type: operations

When an observability feed has missing intervals, first distinguish an ingestion delay from lost events. Compare the collector queue depth, source timestamps, and downstream index lag. This comparison is the same first step regardless of which upstream service produced the feed, and it should be completed before opening any escalation.

A rising queue with old source timestamps points to delayed delivery; an empty queue with absent source records points to loss at the producer. Do not infer a service outage from a dashboard blank spot until the source and ingestion path have been checked.
