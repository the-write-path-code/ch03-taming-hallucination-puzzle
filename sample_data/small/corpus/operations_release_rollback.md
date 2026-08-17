# Operations Runbook: Safe Rollback

Document ID: `OPS-ROLLBACK`
Document type: operations

A rollback is the controlled return to a previously verified release after a deployment causes an unacceptable outcome. Before initiating rollback, confirm that the previous artifact is still available in the release registry and that its configuration has not drifted from what was last verified in staging.

Stop further rollout, preserve the evidence needed for diagnosis, restore the prior artifact, and validate the critical user path. A rollback should not erase logs or remove the failed artifact before investigators have recorded its version and deployment scope.
