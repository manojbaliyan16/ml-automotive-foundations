# TARA — Threat Analysis and Risk Assessment
## Item: Automotive Telematics Control Unit (TCU)
### Standard: ISO/SAE 21434:2021 | Author: Manoj Kumar | Date: June 2026

---

## Overview

This document applies ISO/SAE 21434 TARA methodology to an Automotive Telematics Control Unit — the vehicle's single point of connectivity to the cloud, mobile network, WiFi, and V2X.

The analysis identifies assets, threat scenarios, damage scenarios, risk levels, countermeasures, and open action items following the 7-step TARA process defined in ISO/SAE 21434 Clause 15.

---

## 1. Item Definition and Scope

**Item:** Automotive TCU — Telematics Control Unit

**Purpose:** Bridges the vehicle's internal network to external connectivity (cloud, cellular, WiFi). Enables OTA updates, remote vehicle operations, fleet telemetry, and V2X communication.

**Hardware (generic reference architecture):**
- Application Processor running Linux (connectivity, application logic, cloud communication)
- Microcontroller running AUTOSAR Classic (CAN interface to vehicle network)
- Cellular modem (4G/5G) + GNSS + WiFi/BT
- Supported vehicle networks: CAN (traditional architecture) and Ethernet/SOME-IP (zonal architecture)

**Software stack (relevant to TARA):**
```
Application Layer
  ├── Auth Manager       — fetches cloud auth token
  ├── Remote Operations  — handles remote vehicle commands
  ├── OTA Manager        — manages firmware/app updates
  └── Telemetry Agents   — collects vehicle data

Middleware
  ├── Local MQTT Broker  — in-vehicle pub/sub hub
  ├── MQTT Bridge        — bridges local broker to cloud
  └── Connection Manager — cellular/WiFi routing, firewall

Platform
  ├── Linux (AP side)
  └── AUTOSAR (MCU side — CAN interface)
```

**Scope boundaries:**
- IN scope: OTA update channel, cloud MQTT channel, internal credential management, vehicle wakeup mechanism, CAN data path
- OUT of scope: Vehicle ECU internals, cloud backend infrastructure, cellular network layer

---

## 2. Asset Identification

| # | Asset | Description | Impact if Compromised |
|---|---|---|---|
| A1 | OTA firmware bundle | Application/firmware packages delivered from cloud CDN | CRITICAL — malicious bundle = persistent code execution |
| A2 | Cloud auth token | Authentication credential published on internal MQTT broker | CRITICAL — token = full TCU cloud identity |
| A3 | Cloud MQTT channel | Encrypted connection between TCU and cloud broker | HIGH — all V2C communication compromised |
| A4 | Remote operation commands | Commands for vehicle actuation (unlock, start, locate) | CRITICAL — direct vehicle safety impact |
| A5 | Vehicle wakeup trigger | Remote signal that wakes sleeping vehicle | MEDIUM — availability and unauthorized access |
| A6 | CAN telemetry data | Vehicle sensor data flowing from ECUs through TCU to cloud | MEDIUM — privacy and data integrity |

---

## 3. Threat Scenarios

---

### Threat 1 — OTA Firmware Rollback Attack

**Asset:** A1 — OTA firmware bundle

**Threat scenario:**
Attacker compromises the CDN delivery path or intercepts the OTA download channel. Delivers a legitimately signed firmware bundle with an older version number that contains a known, previously patched vulnerability. The TCU installs it, reintroducing the vulnerability.

**Attack path:**
```
Attacker
  → Compromises CDN or intercepts OTA download (man-in-the-middle)
  → Delivers signed bundle v1.0 (old, vulnerable)
  → OTA Manager installs without version check
  → Known CVE reintroduced into production vehicle
```

**Damage scenario:**

| Category | Impact |
|---|---|
| Safety | LOW — application-level, no direct actuation |
| Financial | MEDIUM — fleet-wide if exploited at scale |
| Privacy | MEDIUM — old vulnerability may enable data exfiltration |
| Operational | HIGH — persistent backdoor in fleet |

**Attack feasibility:** MEDIUM — requires CDN compromise or network interception. Forging a valid signature is infeasible.

**Countermeasures:**

| Countermeasure | Status | Notes |
|---|---|---|
| Bundle signature verification | ✅ Required | Prevents forged bundles |
| Version enforcement — reject downgrade | ✅ Required | Blocks rollback even with valid signature |
| TLS on CDN download channel | ✅ Required | Prevents interception in transit |

**Residual risk:** LOW — signature + version check together make this attack infeasible.

**CAL:** CAL-1

---

### Threat 2 — Cloud MQTT Channel Man-in-the-Middle (MITM)

**Asset:** A3 — Cloud MQTT channel

**Threat scenario:**
Attacker positions between TCU and cloud broker on the cellular or WiFi network. Intercepts MQTT traffic to read sensitive vehicle telemetry or inject unauthorized remote operation commands.

**Attack path:**
```
Attacker
  → Intercepts cellular or WiFi network traffic
  → Attempts TLS session hijack or certificate spoofing
  → Reads vehicle telemetry OR injects remote commands
  → Vehicle responds to injected command (unlock, start)
```

**Damage scenario:**

| Category | Impact |
|---|---|
| Safety | HIGH — injected remote commands reach vehicle actuators |
| Privacy | HIGH — full telemetry stream exposed |
| Financial | HIGH — vehicle theft, liability |

**Attack feasibility:** LOW — mutual TLS requires valid client certificate that attacker cannot obtain.

**Countermeasures:**

| Countermeasure | Status | Notes |
|---|---|---|
| TLS on cloud MQTT channel | ✅ Required | Encrypts all traffic |
| Mutual TLS — client certificate required | ✅ Recommended | Prevents impersonation from either side |
| Certificate pinning | ✅ Recommended | Prevents certificate substitution attacks |

**Residual risk:** LOW — mutual TLS effectively prevents MITM. Certificate pinning adds defense in depth.

**CAL:** CAL-2

---

### Threat 3 — Replay Attack on Remote Operation Commands

**Asset:** A4 — Remote operation commands

**Threat scenario:**
Attacker captures a legitimate cloud-signed remote command (e.g., "unlock vehicle") transmitted over the MQTT channel — possible via endpoint compromise or insider access. Replays the identical command at a later time or location, causing unintended vehicle actuation without the owner's knowledge.

**Attack path:**
```
Attacker
  → Captures valid remote command (TLS endpoint compromise or insider)
  → Stores command payload with valid signature
  → Replays command at target time/location
  → TCU validates signature ✓ — cannot detect replay without freshness check
  → Vehicle actuates (door unlocks, engine starts)
```

**Damage scenario:**

| Category | Impact |
|---|---|
| Safety | HIGH — unintended vehicle actuation |
| Financial | HIGH — vehicle theft enablement |
| Privacy | MEDIUM — location tracking via repeated commands |

**Attack feasibility:** MEDIUM — command capture requires endpoint compromise; replay itself is trivial once captured.

**Countermeasures:**

| Countermeasure | Status | Notes |
|---|---|---|
| Cloud auth token validation | ✅ Required | Verifies command source |
| Message sequence number | ✅ Recommended | Rejects out-of-order or duplicate commands |
| Timestamp + TTL validation | ✅ Recommended | Command expires after N seconds — replay fails |
| Nonce per command | ✅ Best practice | Cryptographically unique — cannot reuse |

**Residual risk:** HIGH if no freshness validation. LOW if sequence number or timestamp/TTL is implemented.

**CAL:** CAL-3

---

### Threat 4 — Auth Token Theft via Lateral Movement

**Asset:** A2 — Cloud auth token

**Threat scenario:**
The Auth Manager fetches the cloud authentication token and publishes it to the internal MQTT broker so all TCU applications can use it. An attacker who compromises any single TCU application (via a vulnerability triggered by malicious external input — CAN frame, MQTT message, SMS payload) can subscribe to the token topic on the internal broker and steal the credential. With the token, the attacker opens a separate cloud connection, fully impersonating the TCU.

**Attack path:**
```
External input
  → Malicious CAN frame / crafted MQTT message / SMS payload
  → Exploits vulnerability in any TCU application
  → Attacker achieves code execution inside TCU process space
  → Subscribes to internal MQTT token topic
  → Obtains valid cloud auth token
  → Opens independent TLS connection to cloud broker
  → Impersonates TCU — sends false telemetry, triggers remote operations
```

**Why this matters:** The cloud MQTT channel (Threat 2) may be perfectly secure with mutual TLS, yet this attack bypasses it entirely. The attacker is already inside the TCU — they don't touch the TLS boundary.

**Damage scenario:**

| Category | Impact |
|---|---|
| Safety | CRITICAL — attacker can trigger any remote operation |
| Privacy | CRITICAL — full access to all vehicle telemetry |
| Financial | CRITICAL — vehicle theft, fleet-wide compromise |

**Attack feasibility:** MEDIUM — requires finding an exploitable vulnerability in any TCU application. Attack surface is large (CAN, MQTT, SMS, HTTP all serve as input vectors).

**Countermeasures:**

| Countermeasure | Status | Notes |
|---|---|---|
| Cloud auth token (scoped credential) | ✅ Required | Limits blast radius per token |
| Internal MQTT broker ACL | ✅ Critical | Only the MQTT Bridge should subscribe to token topic |
| Application sandboxing / process isolation | ✅ Recommended | Compromised app cannot access other processes |
| Input validation on all external data handlers | ✅ Required | CAN, MQTT, SMS, HTTP parsers must be hardened |
| Token rotation / short TTL | ✅ Recommended | Stolen token expires quickly |

**Residual risk:** HIGH without MQTT ACLs. MEDIUM with ACLs but no sandboxing. LOW with full defense-in-depth.

**CAL:** CAL-3

---

## 4. Risk Summary

| # | Threat | Safety Impact | Attack Feasibility | Risk Level | Status |
|---|---|---|---|---|---|
| T1 | OTA firmware rollback | LOW | MEDIUM | **LOW** | ✅ Mitigated — signature + version check |
| T2 | Cloud MQTT MITM | HIGH | LOW | **LOW** | ✅ Mitigated — mutual TLS |
| T3 | Replay attack on remote commands | HIGH | MEDIUM | **HIGH** | ⚠️ Requires freshness validation |
| T4 | Auth token lateral movement | CRITICAL | MEDIUM | **CRITICAL** | ⚠️ Requires MQTT ACL + app sandboxing |

---

## 5. Key Design Recommendations

**For T3 (Replay):** Every remote command must carry a timestamp + TTL or a monotonically increasing sequence number. The TCU must reject any command outside the valid window. This is a standard pattern in automotive remote operations and should be non-negotiable in any production implementation.

**For T4 (Lateral movement):** Defense must be layered:
1. Internal MQTT broker ACLs — strict per-topic, per-client access control
2. Application process isolation — compromised app cannot reach another app's memory or subscriptions
3. Short-lived tokens with rotation — limits the window of exposure even if a token is stolen
4. Input validation hardening — every external data handler (CAN parser, MQTT payload handler, SMS parser) must treat all input as hostile

**General principle:** Security at the network boundary (TLS, mutual auth) is necessary but not sufficient. An attacker who gains access to one component inside the device can bypass perimeter controls entirely. Defense-in-depth inside the device is equally important.

---

## 6. TARA Process Summary (ISO/SAE 21434 Clause 15)

| Step | Completed |
|---|---|
| 1. Define item scope | ✅ Section 1 |
| 2. Identify assets | ✅ Section 2 (6 assets) |
| 3. Define threat scenarios | ✅ Section 3 (4 scenarios) |
| 4. Define damage scenarios | ✅ Per threat — Safety/Financial/Privacy/Operational |
| 5. Assess risk (impact × feasibility) | ✅ Section 4 |
| 6. Define countermeasures | ✅ Per threat |
| 7. Verify countermeasures | ⚠️ T3 and T4 require implementation verification |

---

*Standard: ISO/SAE 21434:2021 — Road Vehicles: Cybersecurity Engineering*
*UN Regulation R155 — Cyber Security Management System (CSMS)*
*Reference architecture based on production automotive telematics systems*
