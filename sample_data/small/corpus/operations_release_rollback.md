# Operations Runbook: Safe Rollback

A rollback is the controlled return to a previously verified release after a deployment causes an unacceptable outcome. Stop further rollout, preserve the evidence needed for diagnosis, restore the prior artifact, and validate the critical user path.

A rollback should not erase logs or remove the failed artifact before investigators have recorded its version and deployment scope.
