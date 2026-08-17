# Orion Switch S-16 Release Notes

Release 3.2.1 corrects an intermittent duplex negotiation issue on copper ports when the connected endpoint advertises Energy Efficient Ethernet. The symptom is link flapping after a power event, not a permanent port failure.

Use change ticket class NET-CHG-208 for staged rollout. The release does not change VLAN configuration or routing policy.
