#!/usr/bin/env python3
"""Generate deterministic synthetic retrieval data for Chapter 3.

Produces long (900-1,600 words), structurally varied synthetic enterprise
documents across five distinct document types to stress-test dense, sparse,
and hybrid retrieval mechanisms.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

PRODUCTS = (
    {
        "name": "Atlas Gateway",
        "prefix": "AX4",
        "kind": "edge gateway",
        "component": "gateway control plane",
        "arch": "ARM64 embedded controller with dual cryptographic co-processors",
    },
    {
        "name": "Nimbus Sensor",
        "prefix": "NSN",
        "kind": "environmental sensor",
        "component": "sensor telemetry service",
        "arch": "low-power multi-spectral optical transducer array",
    },
    {
        "name": "Orion Switch",
        "prefix": "ORS",
        "kind": "network switch",
        "component": "switch management plane",
        "arch": "ASIC-accelerated packet inspection fabric",
    },
    {
        "name": "Helios Reader",
        "prefix": "HRD",
        "kind": "identity reader",
        "component": "reader authentication service",
        "arch": "contactless smart-card interface module with secure enclave",
    },
)

DOCUMENT_TYPES = ("product", "policy", "operations", "pricing", "specification")
REGIONS = ("north", "south", "east", "west", "central", "apac", "emea")
SUPPORT_TIERS = ("Standard", "Priority", "Enterprise", "Mission-Critical")
RETENTION_DAYS = (30, 90, 180, 365, 730, 1095)
STORAGE_TIERS = ("hot", "warm", "cool", "archive", "deep_archive")
FAILURE_MODES = ("keyword_exact_match", "semantic_paraphrase", "long_context", "table_lookup")

PARAPHRASE_PAIRS = (
    ("identity renewal", "certificate rollover"),
    ("optical drift alert", "sensor calibration anomaly"),
    ("egress throttling", "bandwidth saturation control"),
    ("firmware reconciliation", "image consistency verification"),
    ("audit artifact preservation", "compliance log retention"),
    ("failover handover", "secondary node promotion"),
    ("rate limiting trigger", "traffic shaping threshold"),
)

MIN_WORDS = 700


def _word_count(text: str) -> int:
    """Return the word count of a string split by whitespace."""
    return len(text.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic enterprise retrieval corpus."
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=200,
        help="Number of Markdown documents to generate (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic output (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_data/generated_large"),
        help="Destination directory (default: sample_data/generated_large).",
    )
    return parser.parse_args()


def _choose_paragraph(rng: random.Random, sentence_pools: list[list[str]]) -> str:
    """Select one sentence from each pool and join into a cohesive paragraph."""
    return " ".join(rng.choice(pool) for pool in sentence_pools)


def render_product_document(values: dict[str, Any], rng: random.Random) -> str:
    product = values["product"]
    table_metric = values["table_metric_name"]
    table_value = values["table_metric_val"]
    paraphrase_term = values["paraphrase_doc_term"]

    # Section 1: Architecture Overview (2 paragraphs, 5 sentences each)
    p1_1 = _choose_paragraph(
        rng,
        [
            [
                f"The {product['name']} represents a mission-critical deployment in the enterprise edge computing portfolio.",
                f"Engineered for high-reliability operational environments, the {product['name']} provides deterministic real-time data processing capabilities.",
                f"Within modern distributed computing architectures, the {product['name']} functions as a primary connectivity and computation anchor.",
            ],
            [
                f"Its core system architecture relies on an {product['arch']}, specifically designed to isolate real-time telemetry processing from heavy background analytics.",
                f"Hardware acceleration is actively governed by the {product['component']} to maintain sub-millisecond execution deadlines under peak ingress loads.",
                f"System memory subsystems utilize error-correcting code (ECC) registers linked directly to the {product['component']} controller.",
            ],
            [
                f"During regular steady-state operations, node health is continuously monitored by localized diagnostic daemons across the {values['region']} cluster.",
                f"All ingress pipelines rigorously validate payload integrity before dispatching frames downstream to central data aggregation nodes.",
                f"Hardware watchdog circuits are calibrated to reset anomalous execution threads without interrupting physical carrier channels.",
            ],
            [
                f"The system architecture maintains strict physical and logical isolation barriers between control plane signaling and user plane data channels.",
                f"Decoupled bus topologies ensure that peripheral sensor fluctuations do not degrade primary network transport throughput.",
                f"Cryptographic key material is maintained in isolated tamper-evident memory partitions within the host controller.",
            ],
            [
                f"Dynamic thermal throttling algorithms actively modulate co-processor clock frequencies to prevent localized silicon hotspots during high ambient stress.",
                f"Hardware memory access arbitration prioritizes incoming telemetry ingestion queues over background housekeeping daemons.",
                f"Diagnostic telemetry registers capture bus contention intervals to facilitate post-operational latency and throughput analysis.",
            ],
        ],
    )

    p1_2 = _choose_paragraph(
        rng,
        [
            [
                f"To achieve exceptional fault tolerance, redundant microcontrollers supervise power rail stabilization and clock distribution networks throughout the chassis.",
                f"The secondary processing unit remains in hot-standby mode, mirroring internal register states via dedicated high-speed inter-chip buses.",
                f"Input filtering stages employ multi-stage low-pass analog filters combined with digital decimation logic to eliminate high-frequency noise.",
            ],
            [
                f"System boot routines execute a multi-stage cryptographically signed sequence before initializing the {product['component']}.",
                f"Hardware secure boot verifiers ensure that only authenticated kernel images are loaded into execution memory during power-on initialization.",
                f"Non-volatile memory controllers execute wear-leveling algorithms optimized for continuous telemetry logging workflows across industrial flash media.",
            ],
            [
                f"Edge computation modules can dynamically offload batch analytical transformations to adjacent cluster peers when local utilization exceeds eighty percent.",
                f"Distributed consensus agents maintain synchronized state across all active {product['name']} nodes in the local network perimeter.",
                f"Fail-safe fallback routines automatically drop non-essential diagnostic metrics if bus bandwidth falls below certified operational thresholds.",
            ],
            [
                f"In addition, internal power distribution boards incorporate transient voltage suppression arrays to withstand high-energy industrial electrical surges.",
                f"The overall architectural design reflects years of empirical engineering refinement in demanding industrial and enterprise deployments.",
                f"Telemetry serialization pipelines support both compact binary encoding formats and human-readable diagnostic message envelopes.",
            ],
            [
                f"Chassis grounding planes are isolated from digital logic grounds to prevent ground-loop interference across sensitive sensor instrumentation.",
                f"Real-time clock circuits incorporate temperature-compensated crystal oscillators to maintain microsecond timestamp synchronization during network disconnects.",
                f"Multi-layered printed circuit board designs minimize electromagnetic radiation while maintaining optimal signal integrity at high frequencies.",
            ],
        ],
    )

    # Section 2: Compatibility & Protocol Interoperability (2 paragraphs, 5 sentences each)
    p2_1 = _choose_paragraph(
        rng,
        [
            [
                f"Hardware interoperability for the {product['name']} is certified against rigorous IEEE and ISO transport standards.",
                f"Integration boundaries require strict adherence to standard optical and electrical interface communication protocols.",
                f"Downstream subsystems interact with the host chassis through deterministic serialization protocols and verified message structures.",
            ],
            [
                f"The primary component interface interfaces seamlessly with legacy infrastructure across the {values['region']} operating division.",
                f"Upstream link aggregation supports active-passive and active-active failover pairs with sub-second link failure detection.",
                f"Telemetry serialization supports both binary protobuf schemas and standard structured JSON message payloads.",
            ],
            [
                f"When provisioning multi-chassis clusters, engineers must ensure that transceiver modules match the nominal impedance profile.",
                f"Bus contention is mitigated through dynamic arbitration algorithms managed by firmware level microcode routines.",
                f"Inter-chassis synchronization relies on precision time protocol (PTP) beacons distributed across dedicated backplane lines.",
            ],
            [
                f"Secondary interface modules can be hot-swapped during scheduled maintenance windows without taking the primary controller offline.",
                f"Optical transceivers must meet Class 1 laser safety standards and undergo automated calibration prior to link establishment.",
                f"Thermal dissipation across peripheral interface brackets is managed through zoned variable-speed cooling assemblies.",
            ],
            [
                f"Interface transceiver diagnostics query internal EEPROM registers continuously to detect optical degradation before link dropouts occur.",
                f"Signal integrity validation reports must be recorded during commissioning to ensure transmission line reflections remain below standard tolerances.",
                f"Dynamic equalizer filters continuously compensate for high-frequency signal loss over extended twisted-pair copper cabling runs.",
            ],
        ],
    )

    p2_2 = _choose_paragraph(
        rng,
        [
            [
                f"Network engineers deploying the {product['name']} in heterogeneous switching environments must verify VLAN tagging configurations.",
                f"Compatibility testing has validated interoperability with major third-party carrier aggregation switches and enterprise edge routers.",
                f"Jumbo frame encapsulation is supported across all high-speed optical interfaces up to 9000 bytes MTU without fragmentation.",
            ],
            [
                f"Software-defined networking (SDN) controllers can dynamically reconfigure ingress flow tables using standard OpenFlow or NETCONF protocols.",
                f"Legacy asynchronous serial interfaces are supported via auxiliary modular adapters for retrofitting brownfield industrial facilities.",
                f"Electromagnetic compatibility (EMC) shielding prevents cross-talk interference when multi-port cards are operating at maximum signaling rates.",
            ],
            [
                f"All external communication interfaces implement hardware-enforced rate limiting to guard against packet flood denial-of-service conditions.",
                f"Automated protocol negotiation daemons establish optimal baud rates and parity parameters during initial device handshake sequences.",
                f"Diagnostic loopback modes allow remote verification of physical transceivers without requiring physical loopback plugs or on-site staff.",
            ],
            [
                f"Inter-operability matrices are updated quarterly to certify emerging transceiver form-factors and newer optical modulation standards.",
                f"Port negotiation timers can be tuned via administrative CLI commands to prevent flapping when connecting across lossy wireless bridges.",
                f"Quality of service (QoS) mappings preserve IEEE 802.1p priority tags across internal hardware switching queues.",
            ],
            [
                f"Packet buffer management utilizes dynamic memory allocation algorithms to absorb bursty traffic without dropping high-priority control frames.",
                f"Comprehensive interoperability test suites validate proper frame forwarding under high packet collision conditions.",
                f"Cross-connect patch panels should be certified for Cat6A or optical OM4 standards to guarantee full bandwidth utilization.",
            ],
        ],
    )

    # Section 3: Lifecycle Governance & Support Status (2 paragraphs, 5 sentences each)
    p3_1 = _choose_paragraph(
        rng,
        [
            [
                f"Under the {values['support_tier']} support tier, firmware patches and security updates are distributed on a regular monthly cadence.",
                f"Support lifecycle governance adheres strictly to enterprise maintenance guidelines for tier-one operational assets.",
                f"Asset lifecycle status is tracked centrally under ticket class `{values['change_code']}` for corporate compliance verification.",
            ],
            [
                f"End-of-life schedules require a minimum of thirty-six months advance notice prior to the deprecation of hardware revision lines.",
                f"Replacement components are staged in regional spare depots across the {values['region']} territory to meet strict SLAs.",
                f"Security advisory bulletins are automatically synchronized with the corporate vulnerability management framework.",
            ],
            [
                f"Technical escalation paths route directly to tier-three sustaining engineering teams when field diagnostics exceed standard thresholds.",
                f"Firmware rollback provisions ensure that failed field upgrades automatically restore the previously validated gold image.",
                f"Extended warranty provisions cover unexpected component drift and thermal degradation under continuous 24/7 operating conditions.",
            ],
            [
                f"Routine maintenance procedures require field engineers to verify chassis ground resistance before initiating diagnostics.",
                f"Lifecycle telemetry feeds automatically log operational hour totals to estimate MTBF milestones across the fleet.",
                f"Deprecation workflows enforce cryptographic sanitization of non-volatile storage prior to physical asset decommissioning.",
            ],
            [
                f"Hardware revision records and bill-of-materials manifests are archived in engineering databases for twenty years to ensure complete component traceability.",
                f"Field service bulletins are dispatched to certified maintenance partners whenever emerging failure trends are identified.",
                f"Annual support contract reviews evaluate asset performance metrics against agreed operational availability benchmarks.",
            ],
        ],
    )

    p3_2 = _choose_paragraph(
        rng,
        [
            [
                f"Service level agreements dictate that critical security patches must be staged and deployed within fourteen calendar days of official release.",
                f"Routine maintenance windows are coordinated with regional facility managers to minimize disruption to active production lines.",
                f"Customer support portals provide real-time visibility into open support cases, firmware downloads, and certified driver packages.",
            ],
            [
                f"Sustaining engineering teams maintain an extensive hardware test matrix to reproduce reported field anomalies under controlled lab conditions.",
                f"Obsolescence management monitoring tracks silicon component availability from supply chain vendors to anticipate future redesign requirements.",
                f"End-of-service lifecycle transitions include comprehensive migration guides and automated configuration conversion utilities.",
            ],
            [
                f"Quarterly reliability reviews evaluate return-merchandise-authorization (RMA) rates to identify trends in component longevity.",
                f"Field service engineers receive mandatory annual recertification on high-voltage safety and ESD prevention procedures.",
                f"Detailed servicing records are preserved in the central asset database to maintain complete provenance for every deployed chassis.",
            ],
            [
                f"Third-party component supply chains undergo strict counterfeit prevention audits to guarantee the authenticity of replacement modules.",
                f"Automated diagnostic bundles can be generated directly from the administrative console to accelerate support ticket triage.",
                f"Customer-specific patch branches are maintained for mission-critical enterprise environments requiring long-term feature stability.",
            ],
            [
                f"Discontinued models enter an extended maintenance phase during which critical security fixes continue to be provided.",
                f"Hardware recycling and disposal programs comply with international WEEE directives for environmentally responsible decommissioning.",
                f"Dedicated technical account managers review fleet health reports during quarterly operational business reviews.",
            ],
        ],
    )

    # Section 4: Component Topology & Downstream Dependencies (2 paragraphs, 5 sentences each)
    p4_1 = _choose_paragraph(
        rng,
        [
            [
                f"In the overarching distributed network topology, the {product['name']} establishes direct adjacency with upstream aggregation fabrics.",
                f"Physical network links terminate at distributed cross-connect panels managed by the {product['component']} team.",
                f"Downstream sensor endpoints stream raw measurements directly through isolated front-panel interface ports.",
            ],
            [
                f"To maintain routing symmetry, border gateways utilize bidirectional forwarding detection (BFD) over all inter-switch trunks.",
                f"Auxiliary out-of-band management networks provide secure remote console access during core network disruptions.",
                f"Topology discovery protocols periodically broadcast link-state advertisements across all active layer-two segments.",
            ],
            [
                f"Cross-functional teams must ensure that {paraphrase_term} workflows are coordinated with regional network administrators.",
                f"Whenever topology reconfiguration occurs, state synchronization queues must be drained to prevent stale route propagation.",
                f"Redundant power distribution units (PDUs) ensure uninterrupted operation during grid instability or generator failover.",
            ],
            [
                f"Traffic flow prioritization is enforced at ingress buffers using weighted round-robin scheduling algorithms.",
                f"Diagnostic loopback interfaces enable automated remote health probing without interrupting ongoing telemetry collection.",
                f"Edge security policies inspect frame headers to discard unauthorized multicast discovery probes at the port boundary.",
            ],
            [
                f"High-density interconnect cables must be routed along dedicated cable management arms to prevent strain on optical connectors.",
                f"Mesh connectivity models allow dynamic peer-to-peer telemetry re-routing when intermediary aggregation switches undergo maintenance.",
                f"Spanning tree protocol instances are configured with root guard to prevent unauthorized switches from claiming bridge root status.",
            ],
        ],
    )

    p4_2 = _choose_paragraph(
        rng,
        [
            [
                f"Inter-cluster dependency maps illustrate that the {product['name']} serves as the primary time-synchronization server for adjacent edge sensors.",
                f"Downstream data consumers subscribe to message topics managed by the local message brokerage daemon.",
                f"Heartbeat monitoring agents ping adjacent nodes at five-hundred-millisecond intervals to detect link degradation proactively.",
            ],
            [
                f"In the event of an upstream network partition, the device enters local buffer mode, storing telemetry on high-endurance flash arrays.",
                f"When connectivity is restored, an intelligent backpressure mitigation algorithm streams queued records without flooding upstream receivers.",
                f"Dynamic load rebalancing shifts incoming client sessions across available co-processors to maintain uniform processing latency.",
            ],
            [
                f"Physical topology documentation must be updated in the central CMDB within twenty-four hours of any cabling alteration.",
                f"Network segmentation rules enforce micro-perimeters around critical management interfaces to mitigate lateral movement risks.",
                f"Regular architectural audits verify that single-point-of-failure vulnerabilities are eliminated from physical power and data pathways.",
            ],
            [
                f"Telemetry brokers implement distributed consensus protocols to maintain topic metadata consistency across cluster restarts.",
                f"Auxiliary power monitoring daemons report real-time current draw across all chassis slots to prevent PDU overload conditions.",
                f"Dynamic packet filtering rules are synchronized across the edge mesh using secure authenticated control channels.",
            ],
            [
                f"Cluster failover groups are tested semi-annually under simulated load to confirm that failover times remain within SLA targets.",
                f"Downstream client libraries include automated reconnection logic with exponential backoff to handle transient network blips.",
                f"Visual LED indicators on the front panel provide immediate physical verification of link state and cluster membership.",
            ],
        ],
    )

    # Section 5: Specification Table Explanatory Note (1 paragraph, 4 sentences)
    p_spec_intro = _choose_paragraph(
        rng,
        [
            [
                f"The technical specifications summarized in the table below reflect nominal operating values certified during comprehensive hardware validation testing.",
                f"All listed parameters are measured under standard laboratory conditions at an ambient temperature of twenty-five degrees Celsius.",
                f"Deviations from these nominal parameters must be reviewed against the certified operational tolerances specified in official data sheets.",
            ],
            [
                f"Hardware integrators must ensure that power supply ratings and chassis clearance dimensions adhere strictly to the engineering table below.",
                f"Electrical characteristics and interface pinouts conform to industry standard mechanical and electrical form-factor definitions.",
                f"Measurement tolerances are maintained within plus or minus two percent across all calibrated laboratory test benches.",
            ],
            [
                f"System administrators should cross-reference these hardware specifications when planning equipment rack thermal management and power budgeting.",
                f"Values listed in this specification matrix serve as baseline criteria for acceptance testing during initial facility provisioning.",
                f"Any custom hardware modifications void the certified ratings listed in this technical summary table.",
            ],
            [
                f"Detailed physical schematics, pin assignment charts, and connector orientation diagrams are available in the supplementary hardware guide.",
                f"Field engineers must verify that all supply voltages remain within the specified range under both idle and full load conditions.",
                f"Compliance with these physical and electrical parameters is mandatory for maintaining valid warranty coverage.",
            ],
        ],
    )

    # Section 6: Archival Telemetry & Long-Term Retention (2 paragraphs, 5 sentences each)
    p5_1 = _choose_paragraph(
        rng,
        [
            [
                f"Long-term data management dictates that operational telemetry is archived according to regional compliance directives.",
                f"Historical performance baselines provide critical context during quarterly capacity planning and infrastructure audits.",
                f"Operational logs are indexed with synthetic tracking metadata to facilitate cross-system query reconciliation.",
            ],
            [
                f"Under routine audit criteria, the long-term archival rule dictates that data retention for this asset family is strictly set to {values['retention_days']} days in cold storage.",
                f"According to governance stipulations, data retention must be preserved for exactly {values['retention_days']} days within designated storage partitions.",
                f"The compliance baseline enforces that historical audit data retention is maintained for {values['retention_days']} days before automated purging.",
            ],
            [
                f"System operators should log any deviation against maintenance tracking reference `{values['incident_code']}` during review.",
                f"Automated checksum validators verify cold storage archive integrity on a weekly scheduled cron schedule.",
                f"Storage tier migrations transition records from active NVMe pools to dense object storage without modifying document hashes.",
            ],
            [
                f"Disaster recovery runbooks specify that archival mirrors must be synchronized across at least two independent availability zones.",
                f"Audit trail verification certificates are cryptographically countersigned by the central governance authority.",
                f"Emergency retrieval drills are conducted semi-annually to validate recovery time objectives under simulated outage scenarios.",
            ],
            [
                f"Encrypted archive bundles are indexed by date, hardware serial number, and operating territory to support rapid targeted retrieval.",
                f"Storage utilization trends are reviewed quarterly to project future archival capacity requirements across all regions.",
                f"Data retention policies are subject to annual review by the enterprise risk and compliance oversight committee.",
            ],
        ],
    )

    p5_2 = _choose_paragraph(
        rng,
        [
            [
                f"Archival storage buckets utilize client-side encryption algorithms to secure raw telemetry payloads before transmission.",
                f"Metadata catalogs maintain indexing tags spanning timestamp, device identifier, and operating region to accelerate retrospective queries.",
                f"Lifecycle expiration daemons execute purge jobs during off-peak hours to avoid impacting primary database throughput.",
            ],
            [
                f"Compliance officers conduct random sampling of archived records to verify that cryptographic signatures remain mathematically verifiable.",
                f"Data compression algorithms achieve an average reduction ratio of four-to-one on structured sensor telemetry streams.",
                f"Secondary air-gapped backup tapes are rotated off-site on a monthly basis for catastrophic disaster recovery insurance.",
            ],
            [
                f"Retention exception requests require executive authorization from both the legal and information security departments.",
                f"Detailed inventory manifests catalog every stored telemetry partition along with its originating cluster topology details.",
                f"Long-term archival storage health metrics are summarized in monthly infrastructure governance reports.",
            ],
            [
                f"Automated alert policies notify storage administrators when archive partition capacity reaches eighty-five percent of allocated limits.",
                f"Data integrity scrubbers verify block-level hashes across cold storage arrays to eliminate silent data corruption risks.",
                f"Restoration testing exercises confirm that complete archive datasets can be decompressed and loaded within target RTO timeframes.",
            ],
            [
                f"Historical telemetry records older than the retention threshold are securely sanitized following certified Department of Defense procedures.",
                f"Legal hold flags can be applied programmatically to suspend automated purge cycles for designated asset clusters.",
                f"All archival operations generate immutable audit logs that record the identity of the executing process and target dataset.",
            ],
        ],
    )

    # Section 7: Disclaimers (1 paragraph, 4 sentences)
    p6 = _choose_paragraph(
        rng,
        [
            [
                "All parameters, measurements, and operational statements in this document are generated for synthetic testing.",
                "This publication does not disclose proprietary customer data or reflect actual live production network status.",
                "Information presented herein serves exclusively to validate semantic search, hybrid retrieval, and indexing pipelines.",
            ],
            [
                "Field personnel should consult official organizational policy documents before applying configuration changes to production clusters.",
                "Unauthorized reproduction or distribution of synthetic benchmark artifacts is governed by corporate test guidelines.",
                "Technical specifications are subject to revision in accordance with synthetic corpus generation seed updates.",
            ],
            [
                "All enterprise trademarks, product designations, and system components cited in this document are synthetic entities.",
                "No performance guarantee or commercial warranty is expressed or implied by the parameters listed in this benchmark dataset.",
                "This benchmark dataset is authored specifically for technical book demonstrations covering vector search and retrieval optimization.",
            ],
            [
                "Any resemblance to actual commercial hardware specifications or live production network configurations is purely coincidental.",
                "Synthetic corpus parameters are designed to simulate realistic enterprise technical documentation scale and structural complexity.",
                "For inquiries regarding synthetic benchmark generation methodologies, consult the accompanying repository documentation.",
            ],
        ],
    )

    table_lines = [
        "| Specification Parameter | Value | Standard / Unit |",
        "| --- | --- | --- |",
        f"| Product SKU | {values['sku_code']} | Model Reference |",
        f"| Primary Interface | {values['primary_interface']} | Physical Port |",
        f"| Power Consumption | {values['power_consumption']} | Watts (Nominal) |",
        f"| Operating Voltage | {values['operating_voltage']} | Volts AC/DC |",
        f"| Chassis Dimensions | {values['chassis_dimensions']} | mm (W x H x D) |",
        f"| Maximum Throughput | {values['max_throughput']} | Gbps |",
        f"| {table_metric} | {table_value} | Measured Metric |",
    ]
    table_text = "\n".join(table_lines)

    lines = [
        f"# {product['name']} Technical Architecture & System Overview: Record {values['index']:04d}",
        "",
        "## System Metadata",
        "",
        f"- Document Identifier: `{values['fault_code']}`",
        f"- Document Type: {values['document_type']}",
        f"- Operating Region: {values['region'].upper()} Division",
        f"- Primary Subsystem: {product['component']}",
        f"- Support Governance Tier: {values['support_tier']}",
        "",
        "## System Architecture & Processing Pipeline",
        "",
        p1_1,
        "",
        p1_2,
        "",
        "## Hardware Compatibility & Protocol Interoperability",
        "",
        p2_1,
        "",
        p2_2,
        "",
        "## Lifecycle Governance & Support Status",
        "",
        p3_1,
        "",
        p3_2,
        "",
        "## Component Topology & Downstream Dependencies",
        "",
        p4_1,
        "",
        p4_2,
        "",
        "## Technical Specifications Table",
        "",
        p_spec_intro,
        "",
        table_text,
        "",
        "## Archival Telemetry & Long-Term Retention",
        "",
        p5_1,
        "",
        p5_2,
        "",
        "## Regulatory Governance and Disclaimers",
        "",
        p6,
        "",
    ]
    return "\n".join(lines)


def render_policy_document(values: dict[str, Any], rng: random.Random) -> str:
    product = values["product"]
    table_metric = values["table_metric_name"]
    table_value = values["table_metric_val"]
    paraphrase_term = values["paraphrase_doc_term"]

    # Section 1: Policy Scope & Boundary Conditions (2 paragraphs, 5 sentences each)
    p1_1 = _choose_paragraph(
        rng,
        [
            [
                f"This corporate governance policy establishes mandatory operational guardrails for all {product['kind']} assets.",
                f"The purpose of this standard is to ensure consistent risk management across all {product['name']} deployments.",
                f"Enterprise security and operational integrity mandate strict adherence to the boundaries defined in this document.",
            ],
            [
                f"All technical teams within the {values['region']} regional operations division are bound by the compliance rules set forth.",
                f"No system administrator or engineer may deploy or modify {product['component']} instances without explicit authorization.",
                f"This policy applies equally to staging environments, production clusters, and disaster recovery hot sites.",
            ],
            [
                f"The scope encompasses physical hardware configurations, software orchestration layers, and connected peripheral sensors.",
                f"Boundary enforcement is verified continuously via automated configuration compliance daemons.",
                f"Failure to adhere to these operational mandates may result in formal compliance reviews and access revocation.",
            ],
            [
                f"Third-party integrations and contractor workflows must be formally certified under the same security requirements.",
                f"Organizational units must designate an accredited compliance owner for every active product cluster.",
                f"Exemptions must follow the documented exception management protocol before implementation.",
            ],
            [
                f"Operational risk matrices are maintained by the internal controls steering committee to assess compliance maturity.",
                f"Continuous monitoring frameworks provide real-time alerting whenever unauthorized configuration changes are detected.",
                f"All operational policies undergo mandatory annual reassessments to align with evolving regulatory frameworks.",
            ],
        ],
    )

    p1_2 = _choose_paragraph(
        rng,
        [
            [
                f"Compliance verification involves automated static policy analysis and dynamic runtime posture evaluation across all active nodes.",
                f"Policy boundaries are enforced through immutable infrastructure code repositories and automated deployment validation pipelines.",
                f"Any attempted bypass of policy enforcement points generates an immediate high-priority audit log entry in the security console.",
            ],
            [
                f"System configurations that deviate from approved baseline templates are automatically flagged for manual compliance review.",
                f"Departmental managers must certify semi-annually that their team members have reviewed and acknowledged this policy standard.",
                f"Enterprise risk scoring algorithms incorporate policy compliance ratings into overall departmental operational health metrics.",
            ],
            [
                f"Security training modules specifically covering {product['name']} governance are required for all credentialed operators and engineers.",
                f"The scope extends to all virtualized environments, cloud-hosted management planes, and hybrid connectivity fabrics across the enterprise.",
                f"Regular cross-departmental governance forums review policy effectiveness and address emerging operational friction points.",
            ],
            [
                f"All policy amendments must receive formal sign-off from both the chief information security officer and legal counsel.",
                f"Automated compliance test suites are executed as part of the standard continuous integration and deployment pipeline.",
                f"Delegated compliance champions within each business unit facilitate communication and provide guidance on policy interpretation.",
            ],
            [
                f"Breaches of policy standards are classified according to a tiered impact scale that determines mandatory notification timelines.",
                f"Historical policy revisions and justification records are archived permanently in the central compliance document vault.",
                f"Annual third-party audits validate that internal compliance testing methodologies adhere to recognized industry standards.",
            ],
        ],
    )

    # Section 2: Exception Criteria & Risk Escalation Thresholds (2 paragraphs, 5 sentences each)
    p2_1 = _choose_paragraph(
        rng,
        [
            [
                f"Exceptions to standard operational baseline configurations must be submitted under tracking code `{values['change_code']}`.",
                f"Risk acceptance criteria require formal justification when deviating from standard {product['component']} policies.",
                f"Temporary configuration overrides are permitted solely during critical triage phases or approved change windows.",
            ],
            [
                f"Any emergency bypass of security validation filters must be signed off by a lead infrastructure architect within four hours.",
                f"System anomalies classified under `{values['incident_code']}` automatically trigger mandatory risk evaluation procedures.",
                f"Under no circumstances may an exception remain active for longer than fourteen consecutive calendar days without reauthorization.",
            ],
            [
                f"When evaluating boundary case exceptions, reviewers must assess secondary impacts on downstream telemetry streams.",
                f"Risk acceptance documentation must explicitly articulate mitigation controls and rollback procedures.",
                f"All approved deviations are logged to an immutable compliance register for quarterly retrospective analysis.",
            ],
            [
                f"Technical debt incurred from temporary waivers must be tracked in the quarterly engineering backlog.",
                f"Security exceptions involving public network exposure require immediate chief information security officer notification.",
                f"Automated alerting notifies system owners twenty-four hours prior to exception permit expiration.",
            ],
            [
                f"Exception requests that lack clear remediation milestones and designated technical owners will be rejected automatically.",
                f"A risk assessment score must be computed using the standardized enterprise risk rating calculator for each waiver.",
                f"Approved exceptions are published to an internal transparency portal accessible by authorized operational teams.",
            ],
        ],
    )

    p2_2 = _choose_paragraph(
        rng,
        [
            [
                f"Risk severity scoring utilizes a standardized matrix assessing potential financial impact, operational downtime, and security exposure.",
                f"High-risk exceptions require compensatory controls such as enhanced telemetry logging and increased monitoring frequency.",
                f"The exception review committee meets weekly to evaluate pending waiver applications and review active temporary permits.",
            ],
            [
                f"If an active exception causes an unexpected operational incident, the waiver is automatically revoked with immediate effect.",
                f"Historical exception records are maintained in the governance database to identify recurring requests that justify policy revision.",
                f"Emergency waivers granted during off-hours must be ratified during the next business day's formal governance review session.",
            ],
            [
                f"Exception documentation must include a detailed remediation plan outlining how standard compliance will be restored.",
                f"Independent security auditors review all active exceptions during annual compliance assessments to verify valid authorization.",
                f"System administrators who implement unapproved exceptions are subject to formal disciplinary and security review processes.",
            ],
            [
                f"Temporary waivers must specify explicit technical constraints to limit the blast radius of any potential failure.",
                f"The cumulative risk of concurrent active exceptions within a single region is monitored to prevent systemic vulnerability buildup.",
                f"Automated expiration daemons revoke temporary access tokens immediately upon the conclusion of the approved waiver period.",
            ],
            [
                f"Post-waiver verification audits confirm that systems have been successfully restored to standard compliance baseline configurations.",
                f"Exceptions involving third-party vendor software must be accompanied by a vendor-certified remediation commitment letter.",
                f"Executive risk dashboards display real-time counts of active, pending, and expired exceptions across all enterprise domains.",
            ],
        ],
    )

    # Section 3: Governance Hierarchy Explanatory Note (1 paragraph, 4 sentences)
    p_hier_intro = _choose_paragraph(
        rng,
        [
            [
                f"The governance hierarchy matrix detailed below establishes clear authority boundaries and sign-off service level agreements.",
                f"Every operational decision involving {product['name']} infrastructure must follow the approval paths defined in this schedule.",
                f"Escalation targets are designated to ensure timely decision-making during high-priority incidents and emergency changes.",
            ],
            [
                f"Roles within the governance matrix are assigned based on demonstrated competency and verified security clearances.",
                f"Sign-off SLA targets represent maximum allowable turnaround times for reviewing standard and expedited requests.",
                f"Failure to obtain required approvals within the designated timeframe triggers automated escalation to the next management tier.",
            ],
            [
                f"Delegated authority arrangements must be formally documented and approved in advance of planned administrative leaves.",
                f"Approval logs are cryptographically sealed to prevent retroactive tampering or repudiation of governance decisions.",
                f"Regular reviews of the governance role matrix ensure alignment with ongoing organizational restructurings and leadership changes.",
            ],
            [
                f"All designated approvers must complete annual training on risk evaluation methodologies and governance compliance requirements.",
                f"The approval matrix applies equally to software upgrades, hardware reconfigurations, and emergency maintenance interventions.",
                f"Cross-functional sign-offs ensure that security, operations, and architectural perspectives are balanced in every decision.",
            ],
        ],
    )

    # Section 4: Governance Roles & Operational Responsibilities (2 paragraphs, 5 sentences each)
    p3_1 = _choose_paragraph(
        rng,
        [
            [
                f"Governance responsibilities are distributed across operational roles to maintain strict separation of duties.",
                f"Access permissions are governed by role-based access control (RBAC) matrices aligned with the {values['support_tier']} level.",
                f"Operational sign-offs require multi-party consensus to prevent single-point administrative vulnerabilities.",
            ],
            [
                f"Engineers responsible for {paraphrase_term} must ensure that audit logs remain intact throughout the approval cycle.",
                f"Delegated administrators may approve low-risk configuration updates within their designated regional boundaries.",
                f"Cross-regional changes require reciprocal approval from the peer operational director in the destination zone.",
            ],
            [
                f"Emergency escalation paths bypass standard ticket queues to alert on-call incident commanders directly.",
                f"All administrative role assignments are subject to automated thirty-day recertification workflows.",
                f"Privileged service accounts must be restricted to automated tooling and prohibited from interactive shell logins.",
            ],
            [
                f"Dual-custody authorization is enforced for high-impact actions including cryptographic key rotation and data purging.",
                f"Audit trail metadata must capture the originating IP address, timestamp, and identity token for every approval.",
                f"Operational handovers between shifts must include a formal sign-off on all pending and active exceptions.",
            ],
            [
                f"Administrative session timeouts are enforced at fifteen minutes of inactivity to prevent unattended console hijacking.",
                f"Privileged operations require just-in-time privilege elevation that automatically deactivates after task completion.",
                f"Role descriptions and associated privilege boundaries are maintained in the central identity governance catalog.",
            ],
        ],
    )

    p3_2 = _choose_paragraph(
        rng,
        [
            [
                f"Role-based privilege escalation requests require secondary manager approval and automatically expire after eight hours.",
                f"The principle of least privilege governs all user account provisioning across {product['name']} infrastructure.",
                f"Identity federation services validate multi-factor authentication tokens before granting access to administrative consoles.",
            ],
            [
                f"Separation of duties prevents individuals who author configuration policies from approving their own deployment requests.",
                f"Security operations teams perform continuous anomalous behavior detection on all privileged administrative sessions.",
                f"Quarterly entitlement reviews ensure that personnel transfers and departures trigger immediate access deprovisioning.",
            ],
            [
                f"Audit logging daemons capture every administrative keystroke and API invocation for immutable retrospective analysis.",
                f"System custodians are assigned specific operational perimeters to maintain clear lines of accountability across teams.",
                f"Escalation matrices are published on the internal engineering portal and updated after every organizational restructuring.",
            ],
            [
                f"Service accounts used by automation frameworks are audited monthly to ensure credentials rotate according to security policy.",
                f"Temporary contractor accounts are provisioned with strict expiration dates and restricted network egress permissions.",
                f"Privilege assignments are reviewed and re-certified by departmental managers as part of annual compliance reviews.",
            ],
            [
                f"All administrative access events are correlated in real time with the central security information and event management (SIEM) system.",
                f"Violation of role boundaries triggers automated credential suspension and immediate notification to the security operations center.",
                f"Continuous role optimization ensures that access entitlements reflect actual job responsibilities without excess privileges.",
            ],
        ],
    )

    # Section 5: Review Cadence & Compliance Audit Schedule (2 paragraphs, 5 sentences each)
    p4_1 = _choose_paragraph(
        rng,
        [
            [
                f"Internal audit teams conduct comprehensive compliance reviews on a pre-scheduled quarterly cadence.",
                f"Ad-hoc spot audits are initiated whenever security incident alerts exceed predefined regional thresholds.",
                f"System health baselines are cross-referenced against historical metrics to identify gradual configuration drift.",
            ],
            [
                f"Review panels assess adherence to {product['name']} operational standards across all operational regions.",
                f"Non-compliant configurations must be remediated within forty-eight hours of formal audit notification.",
                f"Audit findings are aggregated into executive dashboards to track risk posture trends across the enterprise.",
            ],
            [
                f"Automated compliance scanners run daily differential checks against the approved golden configuration templates.",
                f"Discrepancies identified by automated tooling are automatically converted into remediation tickets for assigned teams.",
                f"Annual third-party compliance assessments validate internal auditing efficacy and procedural consistency.",
            ],
            [
                f"Audit logs and verification artifacts are cryptographically hashed to ensure tamper resistance throughout their lifecycle.",
                f"Remediation timelines may be extended only with written authorization from the enterprise risk committee.",
                f"Repeat findings during consecutive audit cycles result in automatic escalation to executive leadership.",
            ],
            [
                f"Auditors have full read-only access to configuration repositories, deployment pipelines, and operational telemetry archives.",
                f"Audit reports include clear risk ratings and prioritized recommendations for corrective and preventive actions.",
                f"Executive management reviews quarterly audit summaries to ensure appropriate resources are allocated for remediation.",
            ],
        ],
    )

    p4_2 = _choose_paragraph(
        rng,
        [
            [
                f"Audit findings are categorized by severity, with critical issues requiring daily status reporting until full resolution.",
                f"Compliance scorecards are published quarterly to foster healthy competition and accountability among operational regions.",
                f"Internal auditors utilize automated sampling methodologies to inspect at least ten percent of all active device configurations.",
            ],
            [
                f"Audit trail verification includes checking cryptographic timestamp signatures on all recorded operational decisions.",
                f"Remediation verification testing is conducted independently by the internal audit team before closing out findings.",
                f"Annual policy reviews incorporate audit findings and industry best practices to continuously improve the governance standard.",
            ],
            [
                f"Cross-functional retrospective workshops examine recurring compliance friction points to refine automated guardrails.",
                f"Executive management receives a comprehensive annual summary of policy compliance health and risk mitigation progress.",
                f"External regulatory audit packages are prepared and certified using standardized compliance reporting templates.",
            ],
            [
                f"Corrective action plans must specify measurable success criteria and interim verification milestones.",
                f"Audit evidence packages are assembled automatically by compliance daemons to streamline external assessment workflows.",
                f"Continuous auditing tools provide real-time assurance by monitoring key risk indicators across all operational environments.",
            ],
            [
                f"Lessons learned from internal audits are incorporated into mandatory employee training curricula.",
                f"The internal audit charter guarantees organizational independence and unfettered access to all relevant information systems.",
                f"Audit tracking systems maintain complete histories of all findings, recommendations, and management responses.",
            ],
        ],
    )

    # Section 6: Evidence Retention & Verification Artifacts (2 paragraphs, 5 sentences each)
    p5_1 = _choose_paragraph(
        rng,
        [
            [
                f"Mandatory evidence collection standards require comprehensive digital artifact preservation for all system events.",
                f"Audit evidence packages must include raw telemetry logs, signed approval tickets, and automated diff snapshots.",
                f"Compliance teams rely on structured evidence repositories to satisfy external regulatory inquiries.",
            ],
            [
                f"Under the enterprise policy baseline, data retention for audit artifacts is strictly set to {values['retention_days']} days in cold storage.",
                f"Governance mandates dictate that all policy compliance records must maintain a data retention period of {values['retention_days']} days in secure vaults.",
                f"The compliance policy enforces that evidence data retention must span exactly {values['retention_days']} days prior to destruction.",
            ],
            [
                f"Storage partitions containing compliance evidence must be protected by write-once-read-many (WORM) storage policies.",
                f"Chain of custody documentation must accompany any extraction of evidence artifacts for legal or regulatory review.",
                f"Automated lifecycle rules enforce the cryptographic erasure of evidence packages once the retention threshold expires.",
            ],
            [
                f"Periodic integrity verifications perform SHA-256 hash validation across all archived evidence packages.",
                f"Evidence package exports must be watermarked with the requesting investigator's digital signature.",
                f"In the event of an ongoing regulatory inquiry, legal hold flags indefinitely suspend automated retention purges.",
            ],
            [
                f"Evidence storage architectures must guarantee zero bit rot through continuous background data scrub routines.",
                f"Digital signatures on archived evidence packages are periodically re-signed using updated cryptographic algorithms.",
                f"Access to evidence repositories is strictly limited to authorized compliance investigators and internal auditors.",
            ],
        ],
    )

    p5_2 = _choose_paragraph(
        rng,
        [
            [
                f"Automated export tooling packages evidence artifacts into standardized tarballs with self-verifying manifest files.",
                f"Chain-of-custody ledgers record every access, export, and verification event in an append-only cryptographic ledger.",
                f"Disaster recovery plans ensure that evidence repositories are replicated to a geographically distant secondary vault.",
            ],
            [
                f"Decommissioning of evidence storage media follows rigorous physical degaussing and shredding standards.",
                f"Compliance reports confirm that all evidence retention schedules meet statutory and industry regulatory requirements.",
                f"Annual disaster recovery drills validate the rapid restoration and verification of archived evidence packages.",
            ],
            [
                f"Evidence repository metadata catalogs allow rapid search by asset identifier, change ticket, and date range.",
                f"Encryption key management for archived evidence follows strict envelope encryption standards with annual key rotation.",
                f"External legal discovery requests are processed through automated workflows that enforce strict redaction protocols.",
            ],
            [
                f"Storage tier migration daemons ensure evidence records move to cost-effective cold tiers without altering access controls.",
                f"Continuous replication health monitoring triggers immediate alerts if evidence vault replication lag exceeds one hour.",
                f"Compliance officers perform quarterly spot-checks on evidence package integrity to ensure ongoing audit readiness.",
            ],
            [
                f"Final disposition of expired evidence packages requires dual authorization from legal counsel and the chief compliance officer.",
                f"Certificate of destruction records are maintained indefinitely in the corporate compliance register.",
                f"Evidence management procedures undergo annual review to ensure alignment with emerging international data protection standards.",
            ],
        ],
    )

    # Section 7: Disclaimers (1 paragraph, 4 sentences)
    p6 = _choose_paragraph(
        rng,
        [
            [
                "This policy document is a synthetic benchmark resource generated for algorithmic retrieval evaluation.",
                "No actual corporate policy, employee obligation, or regulatory commitment is created by this document.",
                "Content generated within this framework is intended exclusively for benchmark testing in Chapter 3.",
            ],
            [
                "All operational references, product names, and regulatory standards are fictional constructs.",
                "For genuine corporate governance and compliance guidelines, consult internal policy directories.",
                "This document is authored specifically to test dense and sparse retrieval algorithms against structured enterprise policies.",
            ],
            [
                "This synthetic artifact is designed to evaluate dense and hybrid retrieval accuracy in technical book demonstrations.",
                "Any resemblance to existing corporate policies or live enterprise infrastructure is entirely coincidental.",
                "Synthetic governance rules contained herein must not be applied to actual production computing infrastructure.",
            ],
            [
                "The parameters, roles, and SLA commitments listed in this document serve solely as benchmark evaluation tokens.",
                "No warranty or legal representation is made regarding the suitability of these policies for actual compliance use.",
                "For full technical details on synthetic benchmark data generation, refer to the project repository documentation.",
            ],
        ],
    )

    table_lines = [
        "| Governance Level | Approving Role | Review SLA | Escalation Target |",
        "| --- | --- | --- | --- |",
        f"| Tier-1 Standard | Operations Lead | 24 Hours | {values['region'].title()} Regional Director |",
        f"| Tier-2 Elevated | Security Architect | 8 Hours | Principal Security Officer |",
        f"| Tier-3 Critical | VP Infrastructure | 2 Hours | Enterprise Risk Board |",
        f"| {table_metric} | {table_value} | Verified | Policy Authority |",
    ]
    table_text = "\n".join(table_lines)

    lines = [
        f"# Enterprise Policy Standard: {product['name']} Operational Governance (Ref: {values['index']:04d})",
        "",
        "## Policy Metadata",
        "",
        f"- Document Identifier: `{values['fault_code']}`",
        f"- Policy Classification: {values['document_type'].upper()}-COMPLIANCE",
        f"- Jurisdiction: {values['region'].upper()} Operating Division",
        f"- Target Technology: {product['name']} ({product['component']})",
        f"- Compliance Tier: {values['support_tier']}",
        "",
        "## Policy Scope & Boundary Conditions",
        "",
        p1_1,
        "",
        p1_2,
        "",
        "## Exception Criteria & Risk Escalation Thresholds",
        "",
        p2_1,
        "",
        p2_2,
        "",
        "## Approval Hierarchy & Governance Role Matrix",
        "",
        p_hier_intro,
        "",
        table_text,
        "",
        "## Governance Roles & Operational Responsibilities",
        "",
        p3_1,
        "",
        p3_2,
        "",
        "## Review Cadence & Compliance Audit Schedule",
        "",
        p4_1,
        "",
        p4_2,
        "",
        "## Evidence Retention & Verification Artifacts",
        "",
        p5_1,
        "",
        p5_2,
        "",
        "## Policy Authority and Regulatory Disclaimer",
        "",
        p6,
        "",
    ]
    return "\n".join(lines)


def render_operations_document(values: dict[str, Any], rng: random.Random) -> str:
    product = values["product"]
    table_metric = values["table_metric_name"]
    table_value = values["table_metric_val"]
    paraphrase_term = values["paraphrase_doc_term"]

    # Section 1: Pre-Execution Prerequisites (2 paragraphs, 5 sentences each)
    p1_1 = _choose_paragraph(
        rng,
        [
            [
                f"Before executing operational runbooks against the {product['name']}, engineers must verify ambient system stability.",
                f"Standard operating procedures for {product['component']} require strict adherence to pre-execution safety checklists.",
                f"Field engineers must complete all preparatory diagnostics before initiating runtime interventions on {product['name']}.",
            ],
            [
                f"Verify that no active P1 incident alarms exist across the {values['region']} monitoring console before opening tickets.",
                f"Ensure that secondary telemetry links have sufficient buffer capacity to handle temporary logging surges.",
                f"Confirm that all field diagnostic tools have been updated to the latest certified patch level.",
            ],
            [
                f"Validate that emergency contact channels to the {values['support_tier']} on-call bridge are active and responsive.",
                f"Inspect physical cable connections and optical transceiver signal levels prior to commencing maintenance routines.",
                f"Ensure that adjacent nodes in the cluster have sufficient headroom to absorb rerouted traffic during the operation.",
            ],
            [
                f"Check that backup configuration archives have been successfully synchronized to secondary storage nodes.",
                f"Review recent change history logs to ensure no concurrent maintenance activities overlap with this runbook execution.",
                f"Verify that environmental temperature readings in the equipment rack remain within safe operating thresholds.",
            ],
            [
                f"Establish a direct serial console fallback connection in case primary SSH connectivity is lost during the procedure.",
                f"Ensure all team members participating in the runbook execution have reviewed the rollback procedure in advance.",
                f"Confirm that the emergency power shutoff switch is accessible and clearly marked in the equipment room.",
            ],
        ],
    )

    p1_2 = _choose_paragraph(
        rng,
        [
            [
                f"Safety verifications mandate the use of calibrated anti-static wrist straps connected to verified earth ground points.",
                f"Engineers must confirm that replacement hot-swap modules are staged within the data center before initiating tear-down.",
                f"Network capture buffers must be configured with a minimum circular memory allocation of four gigabytes.",
            ],
            [
                f"Pre-flight health checks verify that CPU core temperatures remain below seventy degrees Celsius under baseline load.",
                f"The operational supervisor must formally log the start of the preparation window in the centralized operations dashboard.",
                f"Automated dependency checkers confirm that all upstream database connections are operating in non-blocking mode.",
            ],
            [
                f"All active alarms in the target region must be acknowledged and silenced to prevent notification storming during maintenance.",
                f"Secondary power feeds must be tested under simulated load to guarantee seamless transition in case of utility disruption.",
                f"Communication channels between regional field technicians and central network operations must remain open throughout.",
            ],
            [
                f"Diagnostic toolkits must be verified against current checksums before connecting to production management buses.",
                f"Facility environmental sensors must confirm that ambient humidity levels are within certified non-condensing ranges.",
                f"All participating engineers must sign into the operational bridge channel ten minutes prior to scheduled start time.",
            ],
            [
                f"A snapshot of all active routing tables and interface statistics must be captured and saved to the change ticket record.",
                f"Ensure that appropriate fire suppression system overrides are coordinated with building security if hot work is required.",
                f"Verify that emergency spare fiber optic patch cables are available in the immediate equipment staging area.",
            ],
        ],
    )

    # Section 2: Multi-Branch Diagnostics & Troubleshooting Protocol (2 paragraphs, 5 sentences each)
    p2_1 = _choose_paragraph(
        rng,
        [
            [
                f"When executing multi-branch diagnostics, engineers must isolate whether the fault originates in hardware or software.",
                f"Follow the sequential decision tree below to identify and resolve anomalous behavior in the {product['component']}.",
                f"Systematic fault isolation minimizes service disruption and prevents unnecessary component replacements.",
            ],
            [
                f"If the monitoring dashboard reports error flag `{values['incident_code']}`, immediately initiate branch B telemetry capture.",
                f"For transient telemetry drops, check whether packet retransmissions exceed five percent of total ingress volume.",
                f"When buffer saturation is detected, dynamically increase the queue allocation limit via the management CLI.",
            ],
            [
                f"If interface link flaps occur continuously, swap the optical transceiver and clean the fiber ferrule connection.",
                f"When microcode crashes are suspected, capture a non-volatile memory core dump before initiating a warm restart.",
                f"If latency metrics spike across inter-switch trunks, verify that routing tables have converged without looping.",
            ],
            [
                f"In cases where {paraphrase_term} procedures stall, verify that the local certificate authority endpoint is reachable.",
                f"Always cross-reference diagnostic telemetry against known errata listings before escalating to third-level engineering.",
                f"Document all observed symptoms, diagnostic outputs, and intermediate test results in the central ticket repository.",
            ],
            [
                f"If diagnostic tests fail repeatedly on the primary controller, initiate failover to the secondary standby supervisor.",
                f"Capture all serial console output during reboots to identify kernel panic traces or memory allocation failures.",
                f"Verify power supply input voltages and cooling fan tachometer signals if uncommanded resets occur intermittently.",
            ],
        ],
    )

    p2_2 = _choose_paragraph(
        rng,
        [
            [
                f"Branch A diagnostics focus on optical layer signal integrity and physical transceiver power levels across all ports.",
                f"Branch B diagnostics evaluate memory heap fragmentation, garbage collection pauses, and thread contention metrics in software.",
                f"Branch C diagnostics inspect cryptographic handshake latency and certificate chain validation timeouts on secure channels.",
            ],
            [
                f"When diagnostic routines identify uncorrectable ECC memory errors, the node must be scheduled for immediate physical replacement.",
                f"Automated diagnostic scripts execute fifty synthetic test transactions to verify end-to-end data pipeline integrity.",
                f"If anomalous packet drops persist after resetting the interface, escalate the ticket to sustaining hardware engineering.",
            ],
            [
                f"Diagnostic logs must be captured with microsecond timestamp precision to facilitate retrospective event correlation.",
                f"Engineers should consult the historical knowledge base for known bug patterns matching the observed error signature.",
                f"All diagnostic state changes must be committed to the operational timeline for post-incident review.",
            ],
            [
                f"Branch D procedures cover persistent hardware bus contention and dynamic clock frequency adjustment anomalies.",
                f"Automated diagnostic bundles are exported to the central analysis portal upon completion of the troubleshooting run.",
                f"If thermal sensors report readings above critical thresholds, auxiliary cooling blowers are commanded to maximum speed.",
            ],
            [
                f"Post-diagnostic verification requires all synthetic test probes to complete with zero packet loss over a five-minute period.",
                f"Field technicians must clean and inspect all optical fiber endpoints using video microscope probes before reconnection.",
                f"Diagnostic results are summarized in standard structured format and appended to the permanent asset maintenance history.",
            ],
        ],
    )

    # Section 3: Maintenance Window Constraints & Service Level Impacts (2 paragraphs, 5 sentences each)
    p3_1 = _choose_paragraph(
        rng,
        [
            [
                f"All maintenance operations must be scheduled strictly within designated regional change windows.",
                f"For assets in the {values['region']} zone, the standard change window opens at 02:00 UTC and closes at 05:00 UTC.",
                f"Executing modifications outside approved change windows requires explicit authorization under ticket `{values['change_code']}`.",
            ],
            [
                f"Service level agreements under the {values['support_tier']} tier allow a maximum maintenance disruption of fifteen minutes.",
                f"Customer notifications must be dispatched at least seventy-two hours in advance of any service-impacting procedure.",
                f"Traffic draining procedures must be initiated thirty minutes prior to the start of physical maintenance work.",
            ],
            [
                f"Real-time synthetic transaction probes monitor customer-facing endpoints throughout the entire maintenance window.",
                f"If unforeseen complications arise, the maintenance lead must initiate rollback procedures at least forty-five minutes before window close.",
                f"Post-maintenance verification tests must complete with zero errors before traffic is restored to the primary node.",
            ],
            [
                f"Any overrun past the scheduled change window automatically triggers an operational incident review.",
                f"Change window metrics, including actual downtime and error rates, are recorded for monthly SLA reporting.",
                f"Simultaneous changes to redundant partner nodes are strictly prohibited to maintain cluster availability.",
            ],
            [
                f"Maintenance leads must confirm that all regional support teams have acknowledged the start of the change window.",
                f"Automated status pages are updated at the start and conclusion of the maintenance window to inform downstream users.",
                f"A post-change observation window of thirty minutes is required before the change ticket can be formally resolved.",
            ],
        ],
    )

    p3_2 = _choose_paragraph(
        rng,
        [
            [
                f"Impact analysis assessments must be completed and approved prior to scheduling any high-risk change window.",
                f"During the change window, automated traffic monitors trigger immediate alarms if latency increases by more than ten percent.",
                f"Maintenance bridges must include designated representatives from network engineering, system administration, and customer support.",
            ],
            [
                f"If emergency maintenance is required during business hours, executive approval must be obtained before traffic redirection.",
                f"Post-maintenance soak periods require continuous automated monitoring for a minimum of two hours following restoration.",
                f"Change window completion reports are compiled automatically and emailed to regional stakeholders upon window closure.",
            ],
            [
                f"Customer impact credits are automatically calculated and applied if downtime exceeds contracted service level thresholds.",
                f"Lessons learned from change window execution are reviewed in weekly operational improvement meetings.",
                f"All configuration diffs applied during the window must be committed to the central configuration management repository.",
            ],
            [
                f"Risk categorization dictates whether a change requires on-site engineering presence or can be executed remotely.",
                f"In the event of unexpected network partition during maintenance, pre-configured out-of-band links maintain management access.",
                f"A comprehensive rollback plan must be validated in staging environments before high-impact changes are approved.",
            ],
            [
                f"Change success rates are tracked as a key performance indicator for regional operational engineering teams.",
                f"Automated canary deployments route five percent of production traffic through the modified node before full restoration.",
                f"All stakeholders must participate in the final verification call before the maintenance bridge is officially closed.",
            ],
        ],
    )

    # Section 4: Action Matrix Explanatory Note (1 paragraph, 4 sentences)
    p_act_intro = _choose_paragraph(
        rng,
        [
            [
                f"The diagnostic and remediation action matrix below specifies the exact sequential steps required during runbook execution.",
                f"Each step is governed by strict timeout limits and verifiable success criteria to ensure deterministic operational outcomes.",
                f"Engineers must not proceed to subsequent steps until the current step's verification metric has been completely satisfied.",
            ],
            [
                f"Timeout limits represent hard boundaries; if a step exceeds its allocated timeout, rollback procedures must be evaluated.",
                f"Verification metrics are measured using standardized diagnostic commands executed against the local management controller.",
                f"Step execution timestamps and output values must be recorded in the change ticket for post-operational auditing.",
            ],
            [
                f"Automated runbook orchestration tooling can execute this matrix sequentially while providing real-time telemetry updates.",
                f"Manual execution requires dual-operator verification for all steps categorized as service-impacting or irreversible.",
                f"Any deviation from the prescribed sequence invalidates the runbook execution and requires immediate supervisory notification.",
            ],
            [
                f"Detailed remediation procedures corresponding to each step number are documented in the supplementary operations handbook.",
                f"Field engineers should verify that all required diagnostic tools and software packages are pre-loaded on the target chassis.",
                f"Compliance with this action matrix is audited quarterly as part of standard operational quality assurance reviews.",
            ],
        ],
    )

    # Section 5: Historical Incident Analysis & Post-Mortem Learnings (2 paragraphs, 5 sentences each)
    p4_1 = _choose_paragraph(
        rng,
        [
            [
                f"Historical post-mortem records provide vital context for recurring anomalies in {product['name']} deployments.",
                f"Analysis of previous failure modes indicates that thermal stress frequently correlates with memory register bit flips.",
                f"Reviewing past incident reports helps engineers avoid repeating documented troubleshooting pitfalls.",
            ],
            [
                f"During a previous regional outage, delayed failover execution was traced to stale DNS cache entries on edge nodes.",
                f"A prior review identified that rapid power cycling can corrupt configuration files in non-volatile flash storage.",
                f"Past experience demonstrates that meticulous cable labeling reduces physical troubleshooting time by over forty percent.",
            ],
            [
                f"Post-incident remediation actions have introduced automated watchdog scripts to detect and clear deadlocked threads.",
                f"Lessons learned from past outages emphasize the importance of verifying backup power systems prior to major maintenance.",
                f"Historical telemetry logs serve as a valuable training corpus for automated anomaly detection algorithms.",
            ],
            [
                f"Quarterly post-mortem retrospectives review all high-severity incidents to identify systemic architectural vulnerabilities.",
                f"Engineering teams maintain a searchable repository of post-incident learnings and recommended best practices.",
                f"Recurring operational bottlenecks are prioritized for automated remediation in upcoming software release sprints.",
            ],
            [
                f"Cross-team review meetings examine incident timelines to identify communication bottlenecks and procedural ambiguities.",
                f"Post-incident action items are assigned strict completion deadlines and tracked in executive operational dashboards.",
                f"Anonymized incident summaries are published internally to promote continuous learning across all engineering disciplines.",
            ],
        ],
    )

    p4_2 = _choose_paragraph(
        rng,
        [
            [
                f"Root cause analysis methodologies employ the 'Five Whys' framework to drill down to fundamental systemic weaknesses.",
                f"Corrective action items generated from post-mortems are tracked in the engineering sprint backlog with designated owners.",
                f"Historical failure rate data informs predictive maintenance algorithms that schedule component replacements before failure occurs.",
            ],
            [
                f"Incident post-mortems must be published to the internal engineering wiki within five business days of incident resolution.",
                f"Cross-functional post-mortem reviews encourage blameless culture and transparent exploration of technical failures.",
                f"Telemetry dashboards are continuously updated with custom metric views developed during previous incident investigations.",
            ],
            [
                f"Senior leadership reviews recurring incident patterns to allocate sustaining engineering budget toward architectural refactoring.",
                f"Knowledge transfer sessions share post-mortem insights with frontline support engineers during regular tech-talk meetings.",
                f"Post-incident remediation verification is tracked by the quality assurance team before closing related tracking tickets.",
            ],
            [
                f"Empirical data from past incidents demonstrates that comprehensive automated testing prevents over seventy percent of regressions.",
                f"Incident retrospectives review whether monitoring alerts fired promptly and provided actionable diagnostic information.",
                f"Feedback from incident post-mortems is directly incorporated into updated versions of standard operating procedures.",
            ],
            [
                f"Regional failure rate comparisons highlight the impact of environmental factors such as ambient temperature and power quality.",
                f"Engineering teams conduct blameless game-day simulations to practice incident response under realistic outage conditions.",
                f"Historical MTTR metrics demonstrate consistent improvement following the implementation of standardized runbooks.",
            ],
        ],
    )

    # Section 6: Operational Log Archival & Retention Policy (2 paragraphs, 5 sentences each)
    p5_1 = _choose_paragraph(
        rng,
        [
            [
                f"Operational compliance rules require long-term archiving of all diagnostic traces and runbook execution logs.",
                f"Historical operational telemetry enables long-term reliability trending and predictive failure modeling.",
                f"Runbook execution records are preserved in centralized compliance vaults for audit verification.",
            ],
            [
                f"Under the operational logging standard, data retention for runbook artifacts is strictly set to {values['retention_days']} days in cold storage.",
                f"Engineering standards enforce that all operational trace data retention spans exactly {values['retention_days']} days prior to retirement.",
                f"The operational audit framework mandates a data retention duration of {values['retention_days']} days for all maintenance logs.",
            ],
            [
                f"Archive repositories are synchronized across redundant geographical regions to prevent data loss from site failures.",
                f"Automated verification routines periodically test the readability and integrity of archived diagnostic logs.",
                f"At the conclusion of the retention window, records are securely expunged in accordance with data governance policies.",
            ],
            [
                f"Operational logs containing sensitive diagnostic dumps are encrypted using AES-256 prior to cold storage transfer.",
                f"Engineers may request expedited retrieval of archived logs for historical root cause investigations.",
                f"Retention compliance reports are generated monthly and submitted to the operational governance committee.",
            ],
            [
                f"Log archival pipelines utilize batch compression to reduce cloud storage footprints without loss of diagnostic precision.",
                f"Cryptographic hash manifests accompany each archived log batch to detect any post-archival data corruption.",
                f"Cold storage retrieval workflows are tested quarterly to ensure compliance with disaster recovery SLAs.",
            ],
        ],
    )

    p5_2 = _choose_paragraph(
        rng,
        [
            [
                f"Access permissions for archived operational traces follow the principle of least privilege, requiring explicit ticket justification.",
                f"Retention lifecycle rules automatically transition diagnostic archives from warm tiers to deep freeze storage after ninety days.",
                f"De-identified diagnostic datasets are periodically exported to support machine learning telemetry analysis models.",
            ],
            [
                f"Compliance audits verify that data destruction certificates are issued and archived following automated purge cycles.",
                f"Log storage partitions are monitored for unexpected growth that might indicate runaway logging daemon loops.",
                f"Annual storage architecture reviews optimize indexing strategies to maintain sub-second search speeds on archived metadata.",
            ],
            [
                f"Data integrity scrubbers verify block-level hashes across cold storage arrays to eliminate silent data corruption risks.",
                f"Archived runbook records include full terminal transcripts, command inputs, and system responses for complete provenance.",
                f"Disaster recovery runbooks specify that archival mirrors must be synchronized across at least two independent availability zones.",
            ],
            [
                f"Storage capacity forecasts are updated monthly to ensure sufficient archival quota is provisioned across all regional storage pools.",
                f"Archived log retrieval requests must be authorized by an operations manager and are logged for compliance auditing.",
                f"Legal hold flags can be applied to operational archives to suspend automated expiration during ongoing investigations.",
            ],
            [
                f"Periodic data restoration tests verify that compressed historical archives can be fully restored within designated RTO targets.",
                f"End-of-retention purge cycles execute cryptographic zeroization to ensure complete and unrecoverable data elimination.",
                f"Operational archiving procedures are documented in detail in the enterprise data lifecycle management manual.",
            ],
        ],
    )

    # Section 7: Disclaimers (1 paragraph, 4 sentences)
    p6 = _choose_paragraph(
        rng,
        [
            [
                "This standard operating procedure is a synthetic benchmark document generated for retrieval evaluation.",
                "The instructions, fault codes, and diagnostic steps contained herein are fictional constructs for Chapter 3 testing.",
                "Do not execute commands or procedures described in this document on actual physical hardware or production networks.",
            ],
            [
                "For authentic operational runbooks, engineers must consult the verified internal engineering documentation wiki.",
                "All performance thresholds, timeouts, and metrics are simulated for synthetic corpus generation.",
                "This document is part of a synthetic technical dataset used to benchmark dense and hybrid search capabilities.",
            ],
            [
                "This benchmark document serves to evaluate search relevance and retrieval rankers in technical documentation corpora.",
                "No warranty or operational guarantee is provided for procedures executed outside controlled synthetic test harnesses.",
                "Any operational similarity to live runbook procedures or proprietary hardware architectures is purely coincidental.",
            ],
            [
                "All procedural timings, thresholds, and commands listed herein are designed for benchmark repeatability.",
                "Synthetic operational documents must not be referenced for real-world production incident remediation.",
                "Consult official engineering documentation for certified runtime operating instructions and safety guidelines.",
            ],
        ],
    )

    table_lines = [
        "| Step Number | Diagnostic Action | Timeout Limit | Verification Metric |",
        "| --- | --- | --- | --- |",
        f"| Step 01 | Interface Ping Test | 30 Seconds | 100% Packet Return |",
        f"| Step 02 | Buffer Queue Drain | 120 Seconds | Zero Ingress Drops |",
        f"| Step 03 | Microcode Reload | 300 Seconds | Clean Boot Flag |",
        f"| Step 04 | {table_metric} | {table_value} | Threshold Pass |",
    ]
    table_text = "\n".join(table_lines)

    lines = [
        f"# Standard Operating Procedure: {product['name']} Runtime Operations (Runbook {values['index']:04d})",
        "",
        "## Runbook Metadata",
        "",
        f"- Document Identifier: `{values['fault_code']}`",
        f"- Runbook Category: {values['document_type'].upper()}-OPERATIONS",
        f"- Target Region: {values['region'].upper()} Infrastructure Zone",
        f"- Target Hardware: {product['name']} ({product['component']})",
        f"- SLA Severity Tier: {values['support_tier']}",
        "",
        "## Pre-Execution Prerequisites & Safety Verifications",
        "",
        p1_1,
        "",
        p1_2,
        "",
        "## Multi-Branch Diagnostic & Troubleshooting Protocol",
        "",
        p2_1,
        "",
        p2_2,
        "",
        "## Maintenance Window Constraints & Service Level Impacts",
        "",
        p3_1,
        "",
        p3_2,
        "",
        "## Diagnostic & Remediation Action Matrix",
        "",
        p_act_intro,
        "",
        table_text,
        "",
        "## Historical Incident Analysis & Post-Mortem Learnings",
        "",
        p4_1,
        "",
        p4_2,
        "",
        "## Operational Log Archival & Retention Policy",
        "",
        p5_1,
        "",
        p5_2,
        "",
        "## Operational Sign-off & Runbook Governance",
        "",
        p6,
        "",
    ]
    return "\n".join(lines)


def render_pricing_document(values: dict[str, Any], rng: random.Random) -> str:
    product = values["product"]
    table_metric = values["table_metric_name"]
    table_value = values["table_metric_val"]
    paraphrase_term = values["paraphrase_doc_term"]

    # Section 1: Service Tier Structure & Entitlements (2 paragraphs, 5 sentences each)
    p1_1 = _choose_paragraph(
        rng,
        [
            [
                f"This commercial schedule defines the pricing model and service entitlement tiers for {product['name']} subscriptions.",
                f"Enterprise customers deploying {product['name']} hardware and software choose from a structured tier portfolio.",
                f"The commercial framework outlines fee structures, support commitments, and usage quotas across all subscription levels.",
            ],
            [
                f"Base subscription fees cover access to the {product['component']} orchestration platform and core telemetry APIs.",
                f"Customers in the {values['region']} market receive localized billing support and currency conversion protections.",
                f"Entitlements scale dynamically based on total active device count and aggregate daily message ingestion volume.",
            ],
            [
                f"Standard support plans include business-hours ticketing, while enterprise tiers provide 24/7 dedicated engineering response.",
                f"All subscription plans include automated security patch distribution and baseline firmware upgrade entitlements.",
                f"Hardware replacement guarantees vary by tier, ranging from next-business-day delivery to four-hour on-site courier dispatch.",
            ],
            [
                f"Multi-year subscription commitments unlock discounted annual billing rates and locked price protection guarantees.",
                f"Tier upgrades take effect immediately upon request, with billing prorated for the remainder of the active billing cycle.",
                f"Downgrade requests must be submitted thirty days prior to contract renewal to allow for entitlement adjustments.",
            ],
            [
                f"Dedicated technical account managers are assigned to customers subscribing at the Mission-Critical tier.",
                f"Custom integration consulting services are available on a time-and-materials basis for complex hybrid architectures.",
                f"Billing statements provide granular breakdowns of hardware amortization, software licensing, and cloud ingress costs.",
            ],
        ],
    )

    p1_2 = _choose_paragraph(
        rng,
        [
            [
                f"Subscription contracts include access to the comprehensive developer portal, SDK documentation, and sandbox environments.",
                f"Tier-specific API rate limits ensure predictable performance and prevent noisy-neighbor contention across shared gateways.",
                f"Annual subscription renewals undergo automated account health reviews to recommend optimal tier sizing based on historical usage.",
            ],
            [
                f"Enterprise customers benefit from quarterly executive business reviews covering service reliability and roadmap alignment.",
                f"Licensing agreements grant non-exclusive, worldwide rights to deploy client agent software across authorized endpoints.",
                f"Payment options include corporate credit card billing, wire transfers, and automated electronic funds transfer (EFT).",
            ],
            [
                f"Service level credits are automatically credited to subsequent invoices if monthly availability falls below contracted guarantees.",
                f"Custom service level agreements with sub-hour response times are negotiable under master enterprise purchasing frameworks.",
                f"Billing inquiries are prioritized through a dedicated enterprise finance support desk with dedicated phone and chat channels.",
            ],
            [
                f"Commercial contracts include software escrow provisions to protect enterprise investments in mission-critical deployments.",
                f"License keys are generated and distributed automatically through the enterprise customer licensing portal upon payment verification.",
                f"Annual inflation adjustments are capped at three percent for customers maintaining continuous multi-year commitments.",
            ],
            [
                f"Usage analytics dashboards provide finance teams with granular visibility into department-level resource consumption.",
                f"Subscription tiers can be blended across different operational regions under a single consolidated master enterprise agreement.",
                f"Complimentary onboarding and architecture review sessions are included with all Enterprise and Mission-Critical subscriptions.",
            ],
        ],
    )

    # Section 2: Regional Rate Adjustments & Jurisdiction Rules (2 paragraphs, 5 sentences each)
    p2_1 = _choose_paragraph(
        rng,
        [
            [
                f"Regional pricing structures account for localized infrastructure costs, taxation rules, and telecommunications tariffs.",
                f"Billing rates for deployments within the {values['region']} commercial territory are indexed against regional economic benchmarks.",
                f"Cross-border deployments involving multiple operating regions are billed under the master contract jurisdiction.",
            ],
            [
                f"Value-added tax (VAT), goods and services tax (GST), and applicable customs duties are itemized separately on monthly invoices.",
                f"Currency fluctuation adjustments are evaluated semi-annually and capped at a maximum three percent variance per review.",
                f"Invoices are settled in the designated regional currency specified in the master enterprise agreement.",
            ],
            [
                f"Customers operating in exempt economic zones must submit valid tax exemption certificates prior to invoice generation.",
                f"Regional data sovereignty compliance surcharges may apply to deployments requiring isolated in-country telemetry processing.",
                f"Localized disaster recovery hot-site capacity carries a fifteen percent infrastructure standby surcharge.",
            ],
            [
                f"Payment terms default to net-thirty days from invoice issuance, with late payment penalties accruing at 1.5% monthly.",
                f"Disputed invoice line items must be formally registered within fifteen calendar days of bill presentation.",
                f"Electronic funds transfer (EFT) and automated clearing house (ACH) payments receive a one percent billing rebate.",
            ],
            [
                f"International exchange rate indices are locked at the start of each fiscal quarter to provide predictable budgeting for global enterprises.",
                f"Withholding tax certificates must be submitted electronically to the enterprise billing portal within forty-five days of payment.",
                f"Regional pricing adjustments do not retroactively alter previously executed multi-year fixed-rate commitments.",
            ],
        ],
    )

    p2_2 = _choose_paragraph(
        rng,
        [
            [
                f"Local currency billing options are supported across thirty major jurisdictions to eliminate cross-border foreign transaction fees.",
                f"Subsidiary entities operating under a parent enterprise agreement may inherit the master corporate discount structure.",
                f"Billing disputes do not relieve customers of the obligation to pay undisputed invoice amounts within the agreed credit terms.",
            ],
            [
                f"Custom invoicing formats, including cost-center tagging and purchase order matching, are available for enterprise tier accounts.",
                f"Audit rights allow enterprise customers to verify invoice calculations against raw consumption logs upon thirty days written notice.",
                f"Tax rate updates resulting from statutory legislative changes are applied automatically to affected billing cycles without notice.",
            ],
            [
                f"Customers transitioning between regional billing entities must execute standard novation agreements to transfer active subscriptions.",
                f"Regional support hubs provide localized language support during standard business hours in each primary market territory.",
                f"Cross-regional data transfer costs are bundled into standard subscription fees up to the contracted plan limits.",
            ],
            [
                f"Consolidated invoicing combines billing across multiple subsidiaries and product lines into a single monthly master statement.",
                f"Electronic invoice delivery conforms to international digital invoicing standards including Peppol and Factur-X.",
                f"Credit evaluations are conducted annually for customers requesting extended payment terms exceeding forty-five days.",
            ],
            [
                f"Regional promotional credits and market development incentives are governed by separate commercial addenda.",
                f"Invoicing currency conversions utilize the daily mid-market exchange rates published by the central monetary authority.",
                f"Annual financial reconciliation meetings resolve any outstanding billing discrepancies prior to contract anniversary dates.",
            ],
        ],
    )

    # Section 3: Volume Discount Schedules & Enterprise Terms (2 paragraphs, 5 sentences each)
    p3_1 = _choose_paragraph(
        rng,
        [
            [
                f"Enterprise customers qualifying for high-volume deployments are eligible for tiered volume discounting schedules.",
                f"Discount brackets are calculated based on the total number of provisioned {product['name']} endpoints across all regions.",
                f"Volume discounts apply automatically to recurring subscription fees and high-capacity data ingestion rates.",
            ],
            [
                f"Deployments exceeding one thousand concurrent nodes qualify for custom master service agreement (MSA) negotiations.",
                f"Strategic enterprise partners may negotiate tailored SLA commitments aligned with their mission-critical requirements.",
                f"Volume commitments are reviewed annually, with adjustments applied retroactively if actual deployment exceeds projections.",
            ],
            [
                f"Bundled licensing packages combining {product['name']} with complementary software modules offer additional savings.",
                f"Enterprise agreements may include dedicated technical account management and prioritized feature development requests.",
                f"Early contract renewals extending the agreement term by twenty-four months or more unlock supplementary hardware credits.",
            ],
            [
                f"Volume discounts do not stack with promotional pricing or temporary educational grant credits.",
                f"Subcontractor deployments operating under the customer's tenant qualify toward the aggregate volume calculation.",
                f"Contractual minimum spend commitments must be met annually to maintain negotiated tier discounts.",
            ],
            [
                f"Tiered discount percentages range from five percent for entry volume brackets up to thirty-five percent for global hyperscale deployments.",
                f"Hardware bulk purchases receive volume rebates credited directly against first-year software subscription invoices.",
                f"Commitment ramp-up periods grant new customers a ninety-day grace window to reach targeted volume thresholds without penalty.",
            ],
        ],
    )

    p3_2 = _choose_paragraph(
        rng,
        [
            [
                f"Non-profit institutions and certified academic research facilities qualify for specialized educational discount schedules.",
                f"Reseller partner programs offer margin sharing and co-marketing development funds for certified solutions providers.",
                f"Annual contract true-ups reconcile differences between committed minimum spend and actual consumption metrics.",
            ],
            [
                f"Enterprise agreements include terms governing intellectual property indemnification and mutual non-disclosure covenants.",
                f"Custom engineering development hours can be bundled into multi-year enterprise contracts at preferential hourly rates.",
                f"Volume pricing protections guarantee that annual subscription renewal price increases are capped at the consumer price index.",
            ],
            [
                f"Global purchasing agreements allow multinational corporations to aggregate volume across all international operating entities.",
                f"Strategic alliance partners receive early access to beta software releases and dedicated solution architecture support.",
                f"Termination for convenience clauses in enterprise agreements require ninety days advance written notice and standard pro-rata settlement.",
            ],
            [
                f"Enterprise licensing terms permit flexible redistribution of endpoint licenses across internal business units and subsidiaries.",
                f"Custom compliance reporting and dedicated audit log access are included at no additional charge for volume tier customers.",
                f"Annual customer advisory board participation offers enterprise executives direct input into product development roadmaps.",
            ],
            [
                f"Volume discount tiers are locked for the duration of the multi-year contract term regardless of interim catalog price adjustments.",
                f"Supplementary professional services credits can be utilized for training, migration assistance, or performance optimization.",
                f"Enterprise agreements are governed by standard commercial arbitration procedures to ensure expedited resolution of disputes.",
            ],
        ],
    )

    # Section 4: Master Entitlements Schedule Explanatory Note (1 paragraph, 4 sentences)
    p_ent_intro = _choose_paragraph(
        rng,
        [
            [
                f"The master entitlements and pricing schedule below outlines subscription fees, capacity quotas, and service level commitments.",
                f"All listed subscription fees are quoted in United States Dollars (USD) and exclude applicable regional taxes and customs duties.",
                f"Customers may upgrade their plan tier at any point during the billing cycle to accommodate expanding operational requirements.",
            ],
            [
                f"Plan tiers are designed to scale smoothly from pilot proof-of-concept deployments up to global mission-critical infrastructure.",
                f"Each subscription tier includes a baseline allocation of active endpoints, telemetry ingestion volume, and technical support access.",
                f"Enterprise tier pricing includes custom entitlement sizing and dedicated solution architect consultation hours.",
            ],
            [
                f"Billing cycles default to monthly arrears, with annual upfront billing options offering a ten percent discount on subscription fees.",
                f"Entitlement usage is monitored continuously and summarized in the monthly customer billing statement.",
                f"Modifications to base plan entitlements must be formalized through written amendments to the master service agreement.",
            ],
            [
                f"Technical support response times specified in the schedule represent maximum elapsed time to initial engineer engagement.",
                f"Endpoint allowances apply across all deployed {product['name']} hardware models and software agent instances.",
                f"Compliance with subscription terms is verified through automated license reporting mechanisms integrated into the platform.",
            ],
        ],
    )

    # Section 5: Overage Ingestion Rates & Consumption Metering (2 paragraphs, 5 sentences each)
    p4_1 = _choose_paragraph(
        rng,
        [
            [
                f"Usage exceeding contracted plan limits is metered and billed under transparent overage rate schedules.",
                f"API query volume and telemetry message ingestion are monitored continuously via automated metering gateways.",
                f"Customer administrators receive automated threshold alerts when usage reaches eighty percent of the contracted tier quota.",
            ],
            [
                f"Overage rates for data ingestion are billed per million messages consumed beyond the monthly plan entitlement.",
                f"Burst capacity allowances allow temporary traffic surges up to fifty percent above nominal limits without penalty.",
                f"Sustained overages spanning three consecutive billing cycles prompt a mandatory account review to adjust plan sizing.",
            ],
            [
                f"Unmetered internal diagnostic traffic and {paraphrase_term} handshakes do not count toward commercial consumption quotas.",
                f"Overage charges are itemized on the subsequent monthly billing statement with detailed daily usage breakdowns.",
                f"Customers may purchase prepaid overage credit bundles at a twenty percent discount against standard on-demand rates.",
            ],
            [
                f"Storage overages for high-frequency telemetry archives are calculated based on peak gigabyte usage during the billing cycle.",
                f"Network egress bandwidth consumed during disaster recovery failover drills is exempt from standard overage surcharges.",
                f"Rate limiting is never applied to critical safety telemetry, even when commercial consumption quotas are exceeded.",
            ],
            [
                f"Metering data is aggregated at five-minute intervals and pushed to the billing engine via tamper-evident message pipelines.",
                f"Customers have access to real-time consumption dashboards to track daily message counts and forecast month-end billing amounts.",
                f"Automated alerts can be configured to notify finance teams when projected overage fees exceed custom budgetary thresholds.",
            ],
        ],
    )

    p4_2 = _choose_paragraph(
        rng,
        [
            [
                f"Fair use policies prevent intentional denial-of-service abuse by throttling non-critical diagnostic endpoints during extreme surges.",
                f"Unused monthly quota allowances do not roll over to subsequent billing periods under standard subscription terms.",
                f"Enterprise customers may negotiate custom rollover pools for seasonal business models with variable demand cycles.",
            ],
            [
                f"Detailed consumption logs are exportable via REST APIs to integrate directly with internal corporate cost allocation systems.",
                f"Discrepancies between customer-side telemetry counts and metering gateway logs must be reported within thirty days for audit review.",
                f"Third-party metering certification ensures that billing calculations adhere to recognized weights and measures standards.",
            ],
            [
                f"Automated rate-limiting policies gracefully queue non-urgent analytics queries when tenant quotas are momentarily exhausted.",
                f"Historical consumption patterns are analyzed annually to recommend cost-saving plan optimization strategies.",
                f"Enterprise billing agreements can include annual overage caps to guarantee absolute budget certainty for enterprise finance departments.",
            ],
            [
                f"Real-time usage metering endpoints provide webhooks to trigger customer-side auto-scaling and traffic shaping policies.",
                f"Diagnostic logs generated during support investigations are automatically excluded from customer billable volume calculations.",
                f"Billing dispute resolution procedures guarantee credit adjustments within one billing cycle upon verification of metering errors.",
            ],
            [
                f"Overage rate schedules are guaranteed against unilateral price increases for the duration of the active contract term.",
                f"Customer administrators can set self-imposed hard spending limits to prevent unintended consumption overages.",
                f"Comprehensive consumption analytics reports are delivered monthly alongside standard invoice statements.",
            ],
        ],
    )

    # Section 6: Commercial Ledger Archival & Data Retention (2 paragraphs, 5 sentences each)
    p5_1 = _choose_paragraph(
        rng,
        [
            [
                f"Financial and contract compliance policies dictate that all billing records and invoices are permanently archived.",
                f"Historical billing ledgers provide necessary documentation during corporate financial audits and tax filings.",
                f"Commercial agreement records are maintained in secure electronic vaults to guarantee non-repudiation.",
            ],
            [
                f"Under commercial accounting standards, data retention for pricing and invoice records is strictly set to {values['retention_days']} days in cold storage.",
                f"Financial governance mandates that all billing ledger data retention must span exactly {values['retention_days']} days prior to purge.",
                f"The contract compliance standard enforces a mandatory data retention duration of {values['retention_days']} days for commercial records.",
            ],
            [
                f"Archived commercial records are encrypted with customer-specific keys and stored in geographically separated availability zones.",
                f"Automated compliance tools verify that financial ledgers remain immutable throughout the entire retention lifecycle.",
                f"Upon expiration of the statutory retention period, financial records are expunged in accordance with enterprise data policy.",
            ],
            [
                f"Billing audit logs capture every automated credit application, manual adjustment, and discount override.",
                f"Customer finance teams may request certified copies of historical invoices for tax compliance verification.",
                f"External financial auditors are granted read-only access to archived billing ledgers during annual audit reviews.",
            ],
            [
                f"Financial records archives are safeguarded by write-once-read-many (WORM) storage technologies to prevent alteration.",
                f"Annual financial audits confirm that revenue recognition schedules comply with international financial reporting standards (IFRS 15).",
                f"Cryptographic hash manifests guarantee that archived invoices and payment receipts cannot be retroactively modified.",
            ],
        ],
    )

    p5_2 = _choose_paragraph(
        rng,
        [
            [
                f"Disaster recovery protocols ensure that commercial ledger databases can be fully restored within four hours of a catastrophic event.",
                f"Automated retention monitors notify the corporate comptroller prior to scheduled archival purges to verify no active tax holds exist.",
                f"Historical contract amendments and side-letters are cross-referenced with master agreement records in the digital archive.",
            ],
            [
                f"Certified digital certificates attached to archived invoices confirm authenticity for international tax authority submissions.",
                f"Long-term ledger storage costs are factored into overall platform operational budgets without customer surcharge.",
                f"End-of-life contract archives are securely purged following Department of Defense 5220.22-M media sanitization guidelines.",
            ],
            [
                f"Archive retrieval requests for financial records must follow formal audit authorization protocols to preserve confidentiality.",
                f"Electronic billing ledger archives undergo weekly checksum verification to ensure absolute data fidelity across all storage nodes.",
                f"Statutory tax documentation is maintained in accordance with localized regional accounting retention standards.",
            ],
            [
                f"Financial compliance dashboards track retention milestones and alert governance officers sixty days before scheduled purge dates.",
                f"Auditors verify that billing calculations match contracted tariff schedules across all historical accounting cycles.",
                f"Final certificate of erasure documentation is signed by the head of internal audit upon completion of any authorized ledger purge.",
            ],
            [
                f"Customer finance contacts can download historical billing archives directly from the secure customer portal at any time.",
                f"Immutable audit trails record every ledger query, report export, and administrative verification event.",
                f"Long-term financial records management policies are reviewed annually by external compliance and accounting advisors.",
            ],
        ],
    )

    # Section 7: Disclaimers (1 paragraph, 4 sentences)
    p6 = _choose_paragraph(
        rng,
        [
            [
                "This commercial pricing schedule is a synthetic benchmark document generated for retrieval evaluation.",
                "The rates, tier names, and contract terms listed herein are fictional constructs for Chapter 3 testing.",
                "No actual commercial obligation, billing quote, or binding contract is represented by this synthetic document.",
            ],
            [
                "For genuine commercial pricing and procurement information, contact authorized enterprise sales representatives.",
                "All simulated fees, currency conversions, and discount formulas are generated strictly for benchmark execution.",
                "This synthetic document is authored to test dense and sparse retrieval algorithms against structured commercial terms.",
            ],
            [
                "This document is part of a synthetic technical dataset used to benchmark dense and hybrid search capabilities.",
                "Values contained within this document do not represent real-world market pricing or service commitments.",
                "Any resemblance to existing commercial product pricing or enterprise licensing agreements is entirely coincidental.",
            ],
            [
                "The subscription tiers, fee amounts, and quota limits listed in this schedule serve exclusively as synthetic retrieval evaluation tokens.",
                "No legal or financial liability arises from the terms, conditions, or rates published in this benchmark dataset.",
                "Consult the project repository documentation for complete details on synthetic benchmark data generation methodologies.",
            ],
        ],
    )

    table_lines = [
        "| Subscription Plan Tier | Base Monthly Fee (USD) | Included Endpoints | Ingestion Quota | Support SLA |",
        "| --- | --- | --- | --- | --- |",
        f"| Starter Tier | $499 / month | Up to 25 Devices | 10M Messages/mo | Standard (Next Day) |",
        f"| Professional Tier | $1,499 / month | Up to 100 Devices | 50M Messages/mo | Priority (4 Hours) |",
        f"| Enterprise Tier | $4,999 / month | Up to 500 Devices | 250M Messages/mo | Mission-Critical (1 Hour) |",
        f"| {table_metric} | {table_value} | Custom Quota | Unlimited Scale | Dedicated TAM |",
    ]
    table_text = "\n".join(table_lines)

    lines = [
        f"# Commercial Service Agreement & Pricing Schedule: {product['name']} (Plan {values['index']:04d})",
        "",
        "## Commercial Metadata",
        "",
        f"- Document Identifier: `{values['fault_code']}`",
        f"- Schedule Type: {values['document_type'].upper()}-COMMERCIAL",
        f"- Commercial Territory: {values['region'].upper()} Market Division",
        f"- Licensed Product: {product['name']} ({product['component']})",
        f"- Base Contract Tier: {values['support_tier']}",
        "",
        "## Service Tier Structure & Entitlement Overview",
        "",
        p1_1,
        "",
        p1_2,
        "",
        "## Regional Rate Adjustments & Jurisdiction Rules",
        "",
        p2_1,
        "",
        p2_2,
        "",
        "## Volume Discount Schedules & Enterprise Terms",
        "",
        p3_1,
        "",
        p3_2,
        "",
        "## Master Entitlements & Rate Schedule",
        "",
        p_ent_intro,
        "",
        table_text,
        "",
        "## Overage Ingestion Rates & Consumption Metering",
        "",
        p4_1,
        "",
        p4_2,
        "",
        "## Commercial Ledger Archival & Data Retention",
        "",
        p5_1,
        "",
        p5_2,
        "",
        "## Billing Verification & Contract Terms",
        "",
        p6,
        "",
    ]
    return "\n".join(lines)


def render_specification_document(values: dict[str, Any], rng: random.Random) -> str:
    product = values["product"]
    table_metric = values["table_metric_name"]
    table_value = values["table_metric_val"]
    paraphrase_term = values["paraphrase_doc_term"]

    # Section 1: Interface Protocol & Packet Framing (2 paragraphs, 5 sentences each)
    p1_1 = _choose_paragraph(
        rng,
        [
            [
                f"This technical engineering specification defines physical, electrical, and protocol characteristics of the {product['name']}.",
                f"Engineers and hardware integrators must adhere strictly to the interface standards defined in this specification document.",
                f"The {product['name']} subsystem operates in accordance with rigorous industrial and telecommunications engineering baselines.",
            ],
            [
                f"Protocol framing across the {product['component']} utilizes 64b/66b line encoding with cyclic redundancy checks (CRC-32).",
                f"All ingress data frames must carry a 16-byte preamble containing packet sequence numbers and timestamp metadata.",
                f"Payload serialization enforces big-endian byte ordering across all standardized register structures.",
            ],
            [
                f"Interface handshaking requires continuous carrier detect signals to maintain synchronous physical layer synchronization.",
                f"Maximum transmission unit (MTU) size defaults to 1500 bytes, with configurable jumbo frame support up to 9000 bytes.",
                f"Flow control mechanisms implement IEEE 802.3x pause frames to prevent buffer overflow under peak burst conditions.",
            ],
            [
                f"Differential signaling pairs must be impedance-matched to one hundred ohms with less than ten picoseconds skew.",
                f"Receiver equalization circuits dynamically compensate for high-frequency signal attenuation across PCB traces.",
                f"Bit error rate (BER) performance must remain below 10^-12 under maximum rated throughput conditions.",
            ],
            [
                f"Packet synchronization headers include forward error correction (FEC) blocks to mitigate transmission line bit errors.",
                f"Physical layer transceivers support auto-negotiation to establish optimal signaling rates based on cable quality.",
                f"Hardware state machines transition into safe error recovery states whenever three consecutive frame CRC checks fail.",
            ],
        ],
    )

    p1_2 = _choose_paragraph(
        rng,
        [
            [
                f"Frame delineation logic uses unique delimiter sequences to prevent boundary slippage during continuous bitstreams.",
                f"Header compression algorithms reduce transport overhead by up to twenty percent on recurring telemetry streams.",
                f"Inter-packet gap timing is maintained at a minimum of ninety-six bit times to ensure receiver buffer recovery.",
            ],
            [
                f"Link training sequences execute automatically upon physical cable connection to optimize tap coefficients on equalization filters.",
                f"Logical link control layers multiplex diagnostic command channels alongside high-bandwidth payload streams.",
                f"Interface loopback modes facilitate automated factory testing and remote field diagnostic verification.",
            ],
            [
                f"Hardware packet timestamping achieves sub-nanosecond accuracy utilizing dedicated IEEE 1588 hardware capture registers.",
                f"Frame serialization engines utilize dual-ported circular FIFO buffers to eliminate memory bottleneck contention.",
                f"Signal integrity validation requires eye-diagram mask compliance testing across all operating temperature extremes.",
            ],
            [
                f"Asynchronous serial management interfaces provide out-of-band console access during initial hardware commissioning.",
                f"Protocol parsers incorporate hardware sanity checking to discard malformed frames before buffer allocation.",
                f"Dynamic clock recovery circuits lock onto incoming bitstreams within one hundred microseconds of carrier detection.",
            ],
            [
                f"Diagnostic registers capture detailed transmission line statistics including frame error counts, runt packets, and CRC failures.",
                f"Interface transceiver EEPROM memory stores factory calibration parameters and manufacturer serial number records.",
                f"All high-speed differential signal traces are shielded with interleaved ground vias to minimize electromagnetic radiation.",
            ],
        ],
    )

    # Section 2: Environmental Boundaries & Operating Tolerances (2 paragraphs, 5 sentences each)
    p2_1 = _choose_paragraph(
        rng,
        [
            [
                f"Environmental operating boundaries define the safe operational limits for {product['name']} hardware deployments.",
                f"Operating outside these certified environmental boundaries voids hardware warranties and increases failure probability.",
                f"Continuous environmental telemetry monitoring is mandatory for all production chassis installations.",
            ],
            [
                f"Ambient operating temperature must remain within the certified range of {values['temperature_c']} degrees Celsius nominal baseline.",
                f"Relative humidity levels must be maintained between ten percent and ninety percent, non-condensing.",
                f"Thermal sensors embedded in the chassis trigger automated shutdown if internal core temperatures exceed eighty-five Celsius.",
            ],
            [
                f"Vibration resistance is certified to MIL-STD-810G standards for ground mobile and stationary industrial installations.",
                f"Altitude derating requires maximum operating temperature limits to be reduced by one degree Celsius per three hundred meters above sea level.",
                f"Chassis air filtration assemblies must be inspected and cleaned at ninety-day intervals in industrial environments.",
            ],
            [
                f"Acoustic noise emissions under standard operating loads remain below forty-eight decibels at one meter distance.",
                f"Electrostatic discharge (ESD) immunity conforms to IEC 61000-4-2 Level 4 criteria across all exposed metal surfaces.",
                f"Corrosion resistance is validated through twenty-four-hour salt spray testing per ASTM B117 protocols.",
            ],
            [
                f"Thermal modeling demonstrates that forced-air convection maintains component junction temperatures within certified limits.",
                f"Ingress protection (IP) ratings conform to IP65 standards when installed with certified sealed gasket cable glands.",
                f"Shock tolerance testing subjects the chassis to thirty-G half-sine impact pulses across all three principal axes.",
            ],
        ],
    )

    p2_2 = _choose_paragraph(
        rng,
        [
            [
                f"Atmospheric pressure monitoring triggers early warnings if sudden depressurization occurs in airborne installation pods.",
                f"Ultraviolet (UV) radiation resistance testing verifies that exterior polymer housings resist degradation under direct sunlight.",
                f"Chemical resistance ratings certify resilience against common industrial solvents, lubricants, and hydraulic fluids.",
            ],
            [
                f"Thermal shock testing subjects production units to rapid temperature transitions between -40C and +85C with zero structural failure.",
                f"Condensation sensors detect ambient dew point proximity to activate internal resistive heating elements before moisture forms.",
                f"Chassis ventilation pathways incorporate hydrophobic breathable membranes to equalize internal pressure while repelling liquids.",
            ],
            [
                f"Seismic qualification testing demonstrates structural survival under Zone 4 earthquake acceleration profiles.",
                f"Electromagnetic susceptibility testing confirms immunity to radiated RF fields from 80 MHz to 6 GHz at ten volts per meter.",
                f"Conformal coating applied to internal circuit boards prevents dendritic growth and electrochemical migration under high humidity.",
            ],
            [
                f"Chassis heat sink fins are aerodynamically profiled to prevent dust accumulation in high-particulate industrial environments.",
                f"Thermal imaging surveys during prototype validation confirm uniform heat spreading across power amplifier stages.",
                f"Environmental telemetry is sampled at one-hertz intervals and logged to non-volatile black-box flight recorder memory.",
            ],
            [
                f"Modular fan trays feature hot-swappable dual-ball-bearing blowers with individual speed monitoring and tachometer feedback.",
                f"Chassis grounding studs provide low-impedance bonding points to earth ground to dissipate high-voltage static charges safely.",
                f"Operating altitude limits permit continuous operation up to five thousand meters above mean sea level without auxiliary pressurization.",
            ],
        ],
    )

    # Section 3: Subsystem Interconnect & Electrical Characteristics (2 paragraphs, 5 sentences each)
    p3_1 = _choose_paragraph(
        rng,
        [
            [
                f"Subsystem interconnects provide low-latency communication channels between internal coprocessors and external peripherals.",
                f"The interconnect bus operates at 2.5 GHz clock frequency with dual-edge synchronous data latching.",
                f"Signal integrity across backplane traces is maintained through continuous hardware-driven adaptive filtering.",
            ],
            [
                f"Power distribution networks deliver regulated 3.3V and 1.8V rails with less than twenty millivolts peak-to-peak ripple.",
                f"Transient voltage suppressor (TVS) diodes protect sensitive semiconductor gates against inductive power spikes.",
                f"Galvanic isolation barriers rated to 1500V RMS separate the high-voltage power input from delicate signal lines.",
            ],
            [
                f"Interconnect firmware incorporates automated {paraphrase_term} routines to reconcile communication channel state.",
                f"Clock jitter across the master reference oscillator is constrained to under five picoseconds RMS.",
                f"Thermal coupling between high-power transceivers and sensitive sensor circuitry is minimized via isolated ground planes.",
            ],
            [
                f"Peripheral expansion slots support high-speed PCIe Gen 4 interconnect lanes with automated link negotiation.",
                f"Current limiting protection circuits instantly disconnect faulted peripheral modules to prevent bus-wide collapse.",
                f"Shielded RJ45 and SFP+ receptacles feature integrated magnetics and electromagnetic interference (EMI) gaskets.",
            ],
            [
                f"Power supply efficiency exceeds ninety-two percent under typical operating loads, minimizing wasted heat dissipation.",
                f"Decoupling capacitor arrays situated adjacent to major silicon packages filter high-frequency switching transients.",
                f"Printed circuit board (PCB) layouts utilize twelve-layer FR-4 laminates with solid copper reference ground planes.",
            ],
        ],
    )

    p3_2 = _choose_paragraph(
        rng,
        [
            [
                f"Differential trace routing enforces tight length-matching tolerances within five mils to prevent phase distortion.",
                f"Analog-to-digital conversion circuits employ sixteen-bit successive approximation registers with onboard low-noise reference voltages.",
                f"Overcurrent protection breakers incorporate resettable polymeric positive temperature coefficient (PTC) devices.",
            ],
            [
                f"Impedance discontinuities along transmission lines are characterized using time-domain reflectometry (TDR) during manufacturing quality audits.",
                f"Electromagnetic radiation emission profiles conform to FCC Part 15 Class A and CISPR 32 specifications.",
                f"Backplane connectors utilize gold-plated phosphor bronze contact pins rated for a minimum of five hundred mating cycles.",
            ],
            [
                f"Dynamic power management modes reduce quiescent current draw by sixty percent when peripheral channels are idle.",
                f"Synchronous buck converters operate at 1.2 MHz switching frequencies to minimize inductor physical footprint.",
                f"Internal voltage monitoring circuits trigger automated safety shutdown if supply rails drift by more than five percent.",
            ],
            [
                f"Signal lines routed to external connectors feature ESD suppression diodes rated for 15 kV air discharge.",
                f"Inter-board flex cables utilize polyimide substrates with continuous copper ground shields to prevent crosstalk.",
                f"Board-level power sequencing circuitry ensures that core voltages stabilize before I/O supply rails are energized.",
            ],
            [
                f"High-current power distribution buses utilize thick copper busbars to minimize resistive voltage drop across the backplane.",
                f"Differential signaling receivers feature programmable input termination resistors selectable via microcode registers.",
                f"Electrical stress testing verifies component longevity under continuous maximum rated voltage and current conditions.",
            ],
        ],
    )

    # Section 4: Operating Thresholds Table Explanatory Note (1 paragraph, 4 sentences)
    p_spec_intro = _choose_paragraph(
        rng,
        [
            [
                f"The operating thresholds and engineering tolerances specified in the table below represent certified limits for {product['name']}.",
                f"All listed parameters have been verified through rigorous laboratory stress testing across the full operating temperature envelope.",
                f"Field engineering personnel must ensure that operational parameters remain within the minimum and maximum boundaries defined.",
            ],
            [
                f"Exceeding the maximum tolerance limits listed in this schedule will trigger hardware safety cutoffs and log critical errata.",
                f"Measurement parameters conform to standardized test methodologies defined by international engineering standards organizations.",
                f"Test conditions specify the environmental and electrical context under which nominal values and tolerances were verified.",
            ],
            [
                f"Periodic calibration of test fixtures ensures that measurement precision is maintained across all manufacturing batches.",
                f"Hardware integration guides provide detailed schematics and test point locations for field verification measurements.",
                f"Compliance with these electrical and physical thresholds is required for maintaining certified product safety ratings.",
            ],
            [
                f"Deviations observed during commissioning must be recorded and evaluated by the lead sustaining engineering team.",
                f"Calibration values are pre-programmed into onboard non-volatile memory during factory acceptance testing.",
                f"Detailed tolerance curves for variable environmental conditions are available in the technical appendix.",
            ],
        ],
    )

    # Section 5: Changelog & Revision History (2 paragraphs, 5 sentences each)
    p4_1 = _choose_paragraph(
        rng,
        [
            [
                f"Engineering revision histories document iterative hardware and microcode improvements across manufacturing batches.",
                f"Hardware revision tracking ensures complete traceability of internal component revisions and bill of materials changes.",
                f"Field engineers must verify that hardware revision levels match minimum compatibility matrices prior to module replacement.",
            ],
            [
                f"Revision 2.1 introduced optimized heat sink geometry, reducing core thermal resistance by eighteen percent.",
                f"Revision 2.3 upgraded internal non-volatile memory chips to extend write-cycle durability under continuous telemetry logging.",
                f"Revision 2.4 resolved a race condition in the hardware interrupt handler during simultaneous DMA transfers.",
            ],
            [
                f"Firmware versions are digitally signed using RSA-4096 keys to prevent unauthorized microcode modification.",
                f"Changelog entries are synchronized with internal engineering bug tracking systems under reference `{values['change_code']}`.",
                f"Field upgrade kits include automated verification scripts to validate successful microcode flash operations.",
            ],
            [
                f"Backward compatibility with legacy Revision 1.x chassis is maintained via software-selectable protocol compatibility modes.",
                f"Manufacturing test vectors are archived alongside revision notes to support post-production quality audits.",
                f"Deprecation of older hardware revisions is announced through formal Engineering Change Notifications (ECNs).",
            ],
            [
                f"Revision 3.0 transitioned the internal communication bus to a high-speed serialized ring topology.",
                f"Revision 3.2 incorporated redundant oscillator circuitry to ensure fail-safe clock handover during primary crystal failure.",
                f"Revision 3.4 enhanced ESD protection ratings across all front-panel user interface connectors.",
            ],
        ],
    )

    p4_2 = _choose_paragraph(
        rng,
        [
            [
                f"Engineering change orders (ECOs) are documented in the central product lifecycle management (PLM) system.",
                f"Component obsolescence mitigation updates are validated through comprehensive four-corner environmental stress testing.",
                f"Manufacturing line test fixtures are updated simultaneously with every firmware release to maintain quality control rigor.",
            ],
            [
                f"Customer-reported errata are prioritized in monthly sustaining engineering sprints and tracked via publicly accessible release notes.",
                f"Field modification instructions are distributed to regional support centers whenever critical hardware patches are released.",
                f"Historical revision artifacts, including schematics and Gerber files, are archived in immutable engineering design vaults.",
            ],
            [
                f"Revision 3.6 optimized power converter efficiency, reducing total idle power draw by an additional twelve percent.",
                f"Microcode release notes detail specific bug fixes, performance enhancements, and updated register definition maps.",
                f"Component supplier changes undergo comprehensive qualification testing to confirm identical electrical and thermal performance.",
            ],
            [
                f"Automated regression testing validates hundreds of hardware execution paths before any microcode update is approved for production.",
                f"Field upgrade procedures include dual-bank flash memory verification to guarantee safe recovery in case of power interruption.",
                f"Hardware revision compatibility matrices are published on the engineering support portal for customer reference.",
            ],
            [
                f"Long-term reliability tracking evaluates MTBF improvements across successive hardware revisions in active customer deployments.",
                f"Engineering change notifications provide ninety days advance notice before hardware revisions transition into manufacturing.",
                f"Historical revision documentation is maintained for a minimum of twenty-five years to support long-lifecycle installations.",
            ],
        ],
    )

    # Section 6: Calibration Telemetry & Data Retention Requirements (2 paragraphs, 5 sentences each)
    p5_1 = _choose_paragraph(
        rng,
        [
            [
                f"Engineering compliance frameworks enforce strict data preservation rules for all test and telemetry artifacts.",
                f"Long-term calibration records provide necessary baseline evidence during hardware longevity and drift analyses.",
                f"Historical test logs are indexed with unique asset tags to enable rapid engineering retrospectives.",
            ],
            [
                f"Under the engineering specification standard, data retention for sensor calibration data is strictly set to {values['retention_days']} days in cold storage.",
                f"Specification governance rules enforce that engineering telemetry data retention spans exactly {values['retention_days']} days prior to deletion.",
                f"The technical compliance framework requires a data retention period of {values['retention_days']} days for all calibration logs.",
            ],
            [
                f"Calibration repositories are backed up daily to immutable cloud object storage vaults with continuous checksum monitoring.",
                f"Engineering teams verify data integrity using automated cryptographic hashing prior to quarterly design reviews.",
                f"Archival records older than statutory retention thresholds are securely decommissioned following corporate compliance guidelines.",
            ],
            [
                f"Sensor drift compensation curves derived from historical archives are applied retroactively to baseline datasets.",
                f"Test fixture calibration certificates are linked to individual asset records for audit traceability.",
                f"Engineering change analysis workflows require access to historical telemetry spanning the entire operational lifecycle.",
            ],
            [
                f"Telemetry data compression utilizes lossless delta-encoding to maximize storage density without losing high-frequency sample points.",
                f"Automated calibration verification bots run daily sanity checks across newly archived sensor calibration datasets.",
                f"Engineering repositories maintain detailed provenance metadata including test fixture serial numbers and environmental conditions.",
            ],
        ],
    )

    p5_2 = _choose_paragraph(
        rng,
        [
            [
                f"Access controls for calibration data repositories enforce role-based permissions to prevent unauthorized alteration of baseline profiles.",
                f"Periodic disaster recovery tests verify that archived engineering specifications and calibration matrices can be rapidly recovered.",
                f"End-of-life calibration archives are decommissioned according to certified electronic waste and data destruction standards.",
            ],
            [
                f"Historical telemetry archives provide empirical datasets for training predictive machine learning degradation models.",
                f"Cross-regional synchronization of calibration databases ensures that engineering teams worldwide access identical baseline datasets.",
                f"Retention compliance certificates are generated annually to satisfy international standards organization (ISO 9001) audits.",
            ],
            [
                f"Archival storage partitions are protected with hardware-enforced write-once-read-many (WORM) policies.",
                f"Engineering teams conduct quarterly restoration drills to confirm that historical sensor datasets can be rapidly loaded into analytical tools.",
                f"Data integrity monitoring daemons continuously scan cold storage archives to detect and repair silent block corruption.",
            ],
            [
                f"Legal hold mechanisms can be activated to preserve engineering records during regulatory investigations or patent filings.",
                f"Cryptographic signatures attached to calibration records ensure mathematical proof of data provenance and immutability.",
                f"Storage capacity planning models forecast archival storage expansion needs based on projected telemetry ingestion rates.",
            ],
            [
                f"Decommissioned calibration media is physically destroyed in certified electronic shredding facilities with witnessed destruction certificates.",
                f"Automated lifecycle management rules ensure smooth transition of calibration data across storage tiers over time.",
                f"Historical telemetry analysis contributes directly to the design of next-generation hardware architectures and firmware algorithms.",
            ],
        ],
    )

    # Section 7: Disclaimers (1 paragraph, 4 sentences)
    p6 = _choose_paragraph(
        rng,
        [
            [
                "This technical specification is a synthetic benchmark resource generated for algorithmic retrieval evaluation.",
                "All engineering parameters, tolerances, and pinout descriptions are fictional constructs designed for Chapter 3 testing.",
                "Do not use the electrical or mechanical values in this document for actual physical hardware design or manufacturing.",
            ],
            [
                "For authentic engineering specifications and hardware data sheets, consult certified manufacturer publications.",
                "Simulated values and protocol framing definitions serve exclusively to evaluate semantic search and indexing performance.",
                "This document is authored specifically to test dense and hybrid search capabilities against structured engineering data.",
            ],
            [
                "This specification document is created exclusively to stress-test vector search and hybrid retrieval algorithms.",
                "No physical hardware matching these exact synthetic parameters is manufactured or distributed.",
                "Any resemblance to proprietary hardware specifications or published industry standards is purely coincidental.",
            ],
            [
                "The mechanical dimensions, electrical ratings, and protocol timings in this document serve as synthetic benchmark evaluation tokens.",
                "No technical representation or operational warranty is made regarding the accuracy of the simulated parameters.",
                "Consult the project repository documentation for further details on synthetic benchmark corpus generation.",
            ],
        ],
    )

    table_lines = [
        "| Operating Parameter | Nominal Value | Minimum Limit | Maximum Limit | Test Condition |",
        "| --- | --- | --- | --- | --- |",
        f"| Core Supply Voltage | 3.30 V | 3.15 V | 3.45 V | Full Load |",
        f"| Operating Temperature | {values['temperature_c']} C | -40 C | 85 C | Ambient Air |",
        f"| Clock Frequency | 2.50 GHz | 2.48 GHz | 2.52 GHz | Continuous Wave |",
        f"| {table_metric} | {table_value} | Threshold Min | Threshold Max | Certified Lab |",
    ]
    table_text = "\n".join(table_lines)

    lines = [
        f"# Engineering Interface Specification: {product['name']} Subsystem (Spec {values['index']:04d})",
        "",
        "## Specification Metadata",
        "",
        f"- Document Identifier: `{values['fault_code']}`",
        f"- Specification Class: {values['document_type'].upper()}-SPEC",
        f"- Operating Sector: {values['region'].upper()} Engineering Division",
        f"- Target Hardware Architecture: {product['name']} ({product['component']})",
        f"- Certified Reliability Tier: {values['support_tier']}",
        "",
        "## Interface Protocol & Packet Framing Definitions",
        "",
        p1_1,
        "",
        p1_2,
        "",
        "## Environmental Boundaries & Operating Tolerances",
        "",
        p2_1,
        "",
        p2_2,
        "",
        "## Subsystem Interconnect & Electrical Characteristics",
        "",
        p3_1,
        "",
        p3_2,
        "",
        "## Operating Thresholds & Tolerances Table",
        "",
        p_spec_intro,
        "",
        table_text,
        "",
        "## Changelog & Revision History",
        "",
        p4_1,
        "",
        p4_2,
        "",
        "## Calibration Telemetry & Data Retention Requirements",
        "",
        p5_1,
        "",
        p5_2,
        "",
        "## Standard Compliance & Interoperability Notes",
        "",
        p6,
        "",
    ]
    return "\n".join(lines)


def document_values(index: int, rng: random.Random) -> dict[str, Any]:
    product = dict(rng.choice(PRODUCTS))
    document_type = rng.choice(DOCUMENT_TYPES)
    fault_code = f"{product['prefix']}-{index:04d}"

    # Paraphrase pair selection
    paraphrase_pair = rng.choice(PARAPHRASE_PAIRS)
    # Randomly assign which side is in doc vs which side is queried
    if rng.random() < 0.5:
        doc_term, query_term = paraphrase_pair[0], paraphrase_pair[1]
    else:
        doc_term, query_term = paraphrase_pair[1], paraphrase_pair[0]

    # Unique table metric & value (NOT present in prose)
    table_metric_name = rng.choice(
        (
            "Signal Jitter Threshold",
            "Standby Thermal Dissipation",
            "Peak Burst Bandwidth",
            "Harmonic Distortion Floor",
            "Optical Insertion Loss",
            "Quiescent Current Drain",
            "Dynamic Range Capacity",
            "Carrier Frequency Drift",
        )
    )
    table_metric_val = f"{rng.randint(100, 999)}.{rng.randint(10, 99)} units"

    return {
        "index": index,
        "doc_id": f"{document_type}_{index:04d}_{fault_code.lower()}",
        "product": product,
        "document_type": document_type,
        "region": rng.choice(REGIONS),
        "support_tier": rng.choice(SUPPORT_TIERS),
        "retention_days": rng.choice(RETENTION_DAYS),
        "temperature_c": rng.choice((-40, -20, 0, 5, 25, 40, 55)),
        "incident_code": f"INC-{rng.choice(('P1', 'P2', 'P3'))}-{rng.randint(100, 999)}",
        "change_code": f"CC-{rng.randint(1000, 9999)}",
        "fault_code": fault_code,
        "storage_tier": rng.choice(STORAGE_TIERS),
        "sku_code": f"SKU-{product['prefix']}-{rng.randint(100, 999)}",
        "primary_interface": rng.choice(("100G-QSFP28", "25G-SFP28", "10G-BaseT", "Optical-LC-Duplex")),
        "power_consumption": f"{rng.randint(15, 350)} W",
        "operating_voltage": f"{rng.choice((12, 24, 48, 120, 240))} V",
        "chassis_dimensions": f"{rng.randint(300, 600)}x{rng.randint(40, 180)}x{rng.randint(200, 500)} mm",
        "max_throughput": f"{rng.choice((10, 40, 100, 400))} Gbps",
        "paraphrase_doc_term": doc_term,
        "paraphrase_query_term": query_term,
        "table_metric_name": table_metric_name,
        "table_metric_val": table_metric_val,
    }


def render_document(values: dict[str, Any], rng: random.Random) -> str:
    doc_type = values["document_type"]
    if doc_type == "product":
        doc_content = render_product_document(values, rng)
    elif doc_type == "policy":
        doc_content = render_policy_document(values, rng)
    elif doc_type == "operations":
        doc_content = render_operations_document(values, rng)
    elif doc_type == "pricing":
        doc_content = render_pricing_document(values, rng)
    elif doc_type == "specification":
        doc_content = render_specification_document(values, rng)
    else:
        raise ValueError(f"Unknown document type: {doc_type}")

    words = _word_count(doc_content)
    if words < MIN_WORDS:
        raise AssertionError(
            f"Generated document {values['doc_id']} has {words} words, which is below the minimum threshold of {MIN_WORDS} words."
        )

    return doc_content


def validate_evaluation_record(
    record: dict[str, Any], doc_content: str, values: dict[str, Any]
) -> None:
    """Validate retrieval-stressor invariants programmatically."""
    mode = record["failure_mode"]
    lines = doc_content.splitlines()

    if mode == "table_lookup":
        target_val = values["table_metric_val"]
        # Must appear inside a table row (| ... |)
        table_lines = [l for l in lines if l.strip().startswith("|")]
        non_table_lines = [l for l in lines if not l.strip().startswith("|")]

        in_table = any(target_val in l for l in table_lines)
        in_prose = any(target_val in l for l in non_table_lines)

        if not in_table:
            raise AssertionError(
                f"Table lookup target '{target_val}' not found in any table row of {values['doc_id']}"
            )
        if in_prose:
            raise AssertionError(
                f"Table lookup target '{target_val}' leaked into prose lines of {values['doc_id']}"
            )

    elif mode == "long_context":
        # Target fact is retention_days in cold storage
        target_phrase = f"{values['retention_days']} days"
        matching_line_indices = [
            i for i, l in enumerate(lines) if target_phrase in l and "retention" in l.lower()
        ]
        if not matching_line_indices:
            raise AssertionError(
                f"Long context target phrase '{target_phrase}' not found with 'retention' in {values['doc_id']}"
            )
        # Assert the matching line is in the latter half of the document (line index > 0.40 * total lines)
        earliest_match = matching_line_indices[0]
        total_lines = len(lines)
        if earliest_match < total_lines * 0.40:
            raise AssertionError(
                f"Long context fact appeared too early in {values['doc_id']} (line {earliest_match}/{total_lines})"
            )

    elif mode == "keyword_exact_match":
        fault_code = values["fault_code"]
        occurrences = doc_content.count(fault_code)
        # Should appear in metadata and at most one incidental reference (total <= 3)
        if occurrences < 1 or occurrences > 3:
            raise AssertionError(
                f"Keyword exact match identifier '{fault_code}' appeared {occurrences} times in {values['doc_id']} (expected 1 to 3)"
            )


def evaluation_records(
    items: list[dict[str, Any]], contents: dict[str, str]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for offset, values in enumerate(items[: min(32, len(items))], start=1):
        product = values["product"]
        mode = FAILURE_MODES[(offset - 1) % len(FAILURE_MODES)]
        doc_id = values["doc_id"]
        doc_content = contents[doc_id]

        if mode == "keyword_exact_match":
            query = f"What record contains identifier {values['fault_code']}?"
            notes = "Exact synthetic asset identifier."
        elif mode == "semantic_paraphrase":
            query_term = values["paraphrase_query_term"]
            doc_term = values["paraphrase_doc_term"]
            query = (
                f"Which {values['document_type']} document covers {query_term} "
                f"for {product['name']} in the {values['region']} region?"
            )
            notes = f"The document uses '{doc_term}' rather than '{query_term}'."
        elif mode == "long_context":
            query = (
                f"According to the {values['document_type']} documentation for {product['name']} "
                f"(Record {values['index']:04d}), what is the required data retention period in cold storage?"
            )
            notes = "Retention detail follows multiple sections of distractor text near the end of the document."
        else:
            metric_name = values["table_metric_name"]
            query = (
                f"What is the {metric_name} listed in the table for "
                f"{product['name']} record {values['index']:04d}?"
            )
            notes = "Exact value appears exclusively in a Markdown table."

        record = {
            "id": f"q-{offset:03d}",
            "query": query,
            "expected_doc_ids": [values["doc_id"]],
            "failure_mode": mode,
            "notes": notes,
        }

        # Programmatic invariant validation
        validate_evaluation_record(record, doc_content, values)
        records.append(record)

    return records


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.documents < 1:
        raise SystemExit("--documents must be at least 1.")

    rng = random.Random(args.seed)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    corpus_dir = args.output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    contents: dict[str, str] = {}
    for index in range(1, args.documents + 1):
        values = document_values(index, rng)
        doc_content = render_document(values, rng)
        output_path = corpus_dir / f"{values['doc_id']}.md"
        output_path.write_text(doc_content, encoding="utf-8")
        items.append(values)
        contents[values["doc_id"]] = doc_content

    manifest = {
        "documents": args.documents,
        "seed": args.seed,
        "items": [
            {
                "doc_id": item["doc_id"],
                "path": f"corpus/{item['doc_id']}.md",
                "document_type": item["document_type"],
                "failure_tags": list(FAILURE_MODES),
            }
            for item in items
        ],
    }
    records = evaluation_records(items, contents)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "eval_queries.json", records)
    print(f"Generated {args.documents} documents in {args.output_dir}")
    print(f"Wrote {len(records)} evaluation queries")


if __name__ == "__main__":
    main()
