# Orion Switch S-16 Release Notes

Document ID: `NET-CHG-208`
Document type: product
Product: Orion Switch S-16

Release 3.2.1 corrects an intermittent duplex negotiation issue on copper ports when the connected endpoint advertises Energy Efficient Ethernet. Support tickets on this topic are often opened after a routine firmware audit, since the symptom appears only intermittently and is easy to overlook until a power event triggers link flapping. The underlying negotiation defect predates release 3.2.1 and was traced during a broader review of copper-port link stability across the S-16 product line.

The symptom is link flapping after a power event, not a permanent port failure. Use change ticket class NET-CHG-208 for staged rollout. The release does not change VLAN configuration or routing policy.
