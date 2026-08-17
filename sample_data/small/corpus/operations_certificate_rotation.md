# Operations Runbook: Certificate Rotation

A service certificate nearing expiration should be replaced before its renewal window closes. The practical steps are: generate a new key pair, submit the certificate request, deploy the new chain to a canary instance, validate a client handshake, then expand deployment.

Teams often call this a credential refresh, certificate rollover, or identity renewal. These phrases describe the same operational activity when the certificate is the identity being replaced.
