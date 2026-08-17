# Atlas Gateway X4 Firmware Bulletin

Product: Atlas Gateway X4
Firmware line: 4.8.x

Fault code AX4-E117 means that the secure-element attestation handshake expired before the gateway received a signed response. The code does not indicate a failed network interface. For AX4-E117, confirm that the device clock is synchronized, then rotate the site attestation certificate if its validity period has ended.

A gateway may continue forwarding cached telemetry while in this state, but it must not accept a policy update until attestation succeeds. Do not factory-reset a device solely because AX4-E117 appears.
