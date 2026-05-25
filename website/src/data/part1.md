# DebriSolver - Complete Project Master Document
### Learning Conjunction Dynamics: A Self-Supervised Approach to Satellite Collision Risk Assessment
**SDC2026 · KAU Aerospace Engineering Team · King Abdulaziz University**
*Space Debris Conference 2026 - Saudi Space Agency DebriSolver Competition, Riyadh*

---

> **How to use this document:** This is a living, section-by-section record of everything about this project - from the global problem it addresses, to the research decisions, architecture choices, engineering fixes, results, and lessons learned. Each section is self-contained. Read linearly for full context, or jump to any section independently.

---

## TABLE OF CONTENTS

1. [Background & The Global Problem](#1-background--the-global-problem)
2. [The Competition & Dataset](#2-the-competition--dataset)
3. [Literature Review & Prior Art](#3-literature-review--prior-art)
4. [Our Proposed Solution - The Core Idea](#4-our-proposed-solution--the-core-idea)
5. [Data: From Raw KVN to Structured CSV](#5-data-from-raw-kvn-to-structured-csv)
6. [Feature Engineering & Sequence Preparation](#6-feature-engineering--sequence-preparation)
7. [Model Architecture - The BiGRU](#7-model-architecture--the-bigru)
8. [Training Strategy & Optimization](#8-training-strategy--optimization)
9. [Uncertainty Quantification - MC Dropout](#9-uncertainty-quantification--mc-dropout)
10. [Scoring System - Threat & Confidence](#10-scoring-system--threat--confidence)
11. [The Evaluation Gate - Step 3B](#11-the-evaluation-gate--step-3b)
12. [Production Inference - The Dashboard](#12-production-inference--the-dashboard)
13. [Visualization & Reporting](#13-visualization--reporting)
14. [Software Architecture & Pipeline Design](#14-software-architecture--pipeline-design)
15. [Testing Strategy & Test Suite](#15-testing-strategy--test-suite)
16. [Problems Encountered & How We Solved Them](#16-problems-encountered--how-we-solved-them)
17. [Results & Performance](#17-results--performance)
18. [Libraries, Frameworks & Tools](#18-libraries-frameworks--tools)
19. [Lessons Learned & Future Work](#19-lessons-learned--future-work)
20. [Team & Acknowledgments](#20-team--acknowledgments)
21. [References & Citation](#21-references--citation)

---

## 1. Background & The Global Problem

### 1.1 Low Earth Orbit: A Congested Highway

Low Earth Orbit (LEO) - the band of space roughly between 160 km and 2,000 km altitude - is the most economically and scientifically important region of near-Earth space. It hosts weather satellites, Earth observation platforms, communication constellations (Starlink, OneWeb), GPS infrastructure, the International Space Station, and thousands of scientific instruments. It is also, as of the mid-2020s, critically and dangerously congested.

The congestion did not happen overnight. It is the result of five decades of launches combined with one unavoidable physical fact: **objects in LEO do not simply go away**. In the absence of active deorbiting, a satellite or piece of debris at 600 km altitude will remain in orbit for decades. At 800 km, centuries. At 1,000 km, essentially forever on human timescales.

The numbers as of 2024 are sobering:
- **~27,000 objects** are officially tracked by the US Space Surveillance Network (SSN)
- **~100,000-500,000 objects** smaller than 10 cm are estimated to exist but cannot be tracked
- **~1,000,000+ objects** smaller than 1 cm - each capable of disabling a spacecraft
- **Over 7,000 active satellites** are currently operating in orbit, a number growing by 1,000-2,000 per year as mega-constellations expand

Every launch adds objects. Every collision multiplies them. The Iridium-Cosmos collision of 2009 alone generated over 2,000 trackable debris fragments. This self-reinforcing cascade - where debris creates more debris - is known as the **Kessler Syndrome**, first described by NASA scientist Donald Kessler in 1978. We are not in Kessler Syndrome yet, but some orbital shells are approaching the tipping point.

The consequence for operators is not theoretical. It is daily. Every satellite in LEO is threading a path through a field of fast-moving objects, many of which are invisible to current sensors.

---

### 1.2 What Is a Conjunction Event?

A **conjunction** is any close approach between two objects in orbit - a moment when the predicted separation between them falls below a defined screening threshold, typically several kilometers. Most conjunctions are harmless: the objects pass at distances that pose no real danger. But predicting exactly how close two objects will come requires knowing their precise positions and velocities, and orbital mechanics is a discipline humbled daily by measurement noise, atmospheric drag uncertainty, and solar weather.

The standard product of conjunction analysis is the **Conjunction Data Message (CDM)** - a structured document, standardized by the Consultative Committee for Space Data Systems (CCSDS), that reports:

- **TCA (Time of Closest Approach):** The predicted moment of closest pass
- **MISS_DISTANCE:** The predicted closest separation in meters
- **COLLISION_PROBABILITY (Pc):** A probabilistic estimate of the chance of collision, ranging from near-zero to, in extreme cases, several percent
- **Covariance matrices** for both objects: quantifying the uncertainty in each object's position and velocity
- **State vectors:** The position (X, Y, Z) and velocity (Ẋ, Ẏ, Ż) of each object at a reference epoch

CDMs are not issued once per conjunction event. They are issued repeatedly - sometimes every few hours - as new tracking data becomes available and orbital predictions are refined. A conjunction that appears alarming on Monday (Pc = 1×10⁻³) may appear benign by Thursday (Pc = 1×10⁻⁷) as fresh radar tracks tighten the uncertainty. Or it may grow more alarming. This temporal evolution - the **trajectory of Pc over time** - is the signal our system learns to interpret.

The standard reference thresholds used by ESA, NASA, and most commercial operators are:
- **Pc > 1×10⁻⁴** (1-in-10,000): Maneuver evaluation threshold - operator begins seriously considering an avoidance maneuver
- **Pc > 1×10⁻³** (1-in-1,000): High concern - maneuver likely executed unless additional data lowers the risk
- **Pc < 1×10⁻⁶** (1-in-1,000,000): Typically deprioritized - routine monitoring only

---

### 1.3 The Alert Overload Problem

Here is the operational reality that defines the problem: **satellite operators receive thousands of conjunction alerts per year, per satellite.**

A constellation operator managing 500 satellites may receive 50,000-200,000 conjunction screening alerts annually. Each alert nominally demands human attention. Each demands a decision: is this worth examining in detail? Does it require a maneuver? Do we need more tracking data?

The screening tools that generate these alerts are necessarily conservative. The cost of a missed real conjunction (satellite loss, debris cascade, potential casualties on the International Space Station) far exceeds the cost of a false alarm (analyst time, unnecessary maneuver fuel burn). So the systems are tuned to flag broadly, and they do.

The result is **alert fatigue** - a well-documented phenomenon in safety-critical domains where operators are overwhelmed by the volume of alerts, leading to desensitization, missed real threats, and poor decision quality. Space operations is no exception. Human analysts simply cannot give careful attention to every alert.

Compounding this: tracking data quality is uneven. Different sensors - ground-based radar, optical telescopes, TLE-based propagators - produce CDMs with very different uncertainty characteristics. A CDM from a high-precision radar track is fundamentally more trustworthy than one from an old Two-Line Element (TLE) set. But raw Pc values don't tell the operator which is which. A Pc of 1×10⁻⁵ from a well-tracked object and a Pc of 1×10⁻⁵ from a poorly-tracked one demand very different responses. The first might be safely deprioritized. The second demands urgent additional tracking.

---

### 1.4 The Cost of Getting It Wrong

Every conjunction alert forces a decision that carries costs in both directions:

**The cost of acting unnecessarily (false positive):**
- Fuel expenditure - a satellite's lifetime is measured in kilograms of propellant. Each unnecessary avoidance maneuver consumes a finite, irreplaceable resource
- Operational disruption - maneuvers require attitude changes, communication windows, and coordination with ground stations. They interrupt normal operations
- Opportunity cost - scientific instruments may be powered off, imaging opportunities missed, data collection interrupted
- Risk introduction - paradoxically, a poorly-timed maneuver can increase collision probability with a *different* object by changing the orbital trajectory

**The cost of not acting (false negative):**
- **Total loss of a spacecraft** valued at hundreds of millions to billions of dollars
- Loss of irreplaceable scientific data and infrastructure
- Potential danger to crew on the ISS if debris affects their orbit
- Generation of a new debris cloud - the Iridium-Cosmos 2009 collision, between two spacecraft that together massed roughly 1,500 kg, generated over 2,000 tracked debris fragments that will remain hazardous for decades

The operational goal is not to minimize one type of error - it is to make **correctly calibrated decisions** that account for both risk and confidence. An operator who knows a conjunction has high threat *and* high confidence can act decisively. An operator who knows the confidence is low can request more tracking data rather than acting on noisy information. This distinction - between the threat level and the certainty about the threat - is the core of what DebriSolver provides.

---

### 1.5 The Fundamental Data Challenge: No Collision Labels

Every conventional machine learning approach to collision risk assessment runs into the same wall: **there are almost no labeled examples of actual collisions.**

In the entire history of spaceflight, only a handful of orbital collisions have been documented:
- **Iridium 33 vs Cosmos 2251 (February 10, 2009):** The only confirmed hypervelocity collision between two intact spacecraft. Generated ~2,300 tracked debris fragments.
- **FengYun-1C deliberate destruction (January 11, 2007):** China's anti-satellite test created the largest debris cloud in history - over 3,000 tracked fragments, many of which will remain in orbit for decades.
- A handful of smaller collisions involving derelict satellites and debris fragments

That is it. Two events (plus the ASAT test) that could serve as "positive" training examples in a supervised classification framework. Every other conjunction - hundreds of thousands of them - resolved without collision.

This is not a dataset problem that can be solved with more data collection. Actual collisions are rare by design - the entire orbital mechanics community works to prevent them. The class imbalance is not 10:1 or 100:1; it is closer to **1,000,000:1** (successful close approaches to actual collisions).

A supervised classifier trained on this data would learn one thing: predict "no collision" always. It would achieve 99.9999% accuracy while being completely useless.

This is why our work is built on a **self-supervised formulation** that never requires collision labels. The model learns from the dynamics of CDM sequences - how Pc, miss distance, and covariance evolve over time - without ever needing to know whether the event eventually resulted in a collision.

---

### 1.6 Why Existing Systems Are Insufficient

Current conjunction screening systems share a set of structural limitations:

**1. They are threshold-based, not learning-based.**
Most operational systems issue alerts when Pc crosses a fixed threshold (e.g., 1×10⁻⁴). This treats every alert as binary and ignores the temporal trajectory of Pc. An event with Pc = 2×10⁻⁴ and falling (from 5×10⁻³ last week) is very different from one at 2×10⁻⁴ and rising (from 1×10⁻⁶ yesterday). Threshold-based systems cannot distinguish them.

**2. They provide no measure of confidence in their own estimates.**
A Pc value from a CDM carries implicit uncertainty - the uncertainty encoded in the covariance matrices - but this uncertainty is rarely communicated to operators in actionable form. The covariance matrices are large, hard to interpret, and not summarized into a single confidence signal.

**3. They do not learn from historical patterns.**
The behavior of conjunctions across thousands of events contains learnable patterns: how Pc typically evolves as TCA approaches, what covariance evolution looks like for well-tracked vs. poorly-tracked objects, what trajectories typically resolve safely. Rule-based systems ignore this accumulated knowledge.

**4. They treat each CDM in isolation.**
Most existing alert systems evaluate each CDM independently. They do not model the *sequence* of CDMs as a temporal signal. The story told by the evolution of 10 CDMs over a week is far richer than any single CDM snapshot.

**5. They cannot distinguish data quality from physical risk.**
A high Pc from a poorly-tracked object (large covariance) may be entirely an artifact of tracking uncertainty. A lower Pc from a well-tracked object may actually represent more genuine risk. Existing threshold-based systems conflate these two very different situations.

Our system addresses all five of these limitations directly. It learns temporal patterns from CDM sequences (addressing 1 and 4), produces an explicit confidence signal (addressing 2 and 5), and does so by learning from historical conjunction data at scale (addressing 3).

---

## 2. The Competition & Dataset

### 2.1 Space Debris Conference 2026 (SDC2026)

The **Space Debris Conference 2026 (SDC2026)** was organized by the **Saudi Space Agency (SSA)** and held in Riyadh, Saudi Arabia on January 26-27, 2026. It brought together space agencies, commercial operators, researchers, and universities to address one of the most pressing challenges in modern spaceflight: the growing threat of orbital debris.

Alongside the conference, the SSA launched the **DebriSolver Competition** - a technical challenge inviting teams to develop novel, data-driven approaches to the conjunction risk assessment problem. The competition was open to university teams and research groups, providing real CDM data from the French space surveillance company **ALDORIA** as the training and evaluation corpus.

The Saudi Space Agency's motivation for organizing this event reflects Saudi Arabia's growing ambition in the space sector. King Abdulaziz University (KAU), the Kingdom's oldest and one of its most prestigious research universities, entered the competition through its **Aerospace Engineering Department**, fielding a team of five engineers with backgrounds spanning machine learning, orbital mechanics, and systems engineering.

The competition was not structured as a Kaggle-style leaderboard with a fixed metric. Instead, teams were evaluated holistically on the quality and novelty of their approach, the rigor of their technical methodology, and the operational relevance of their solution. This framing encouraged creative approaches - which is why a self-supervised learning system was viable as a submission.

---

### 2.2 The DebriSolver Challenge Brief

The challenge asked participants to address a deceptively simple question: **given a stream of Conjunction Data Messages about a potential satellite collision, how should an operator prioritize their response?**

More precisely, the challenge required teams to:

1. **Ingest and process CDM data** in the industry-standard KVN (Keyhole Notation) format
2. **Develop a method** for assessing the credibility and urgency of conjunction alerts
3. **Produce actionable outputs** that an operator could use to triage hundreds of simultaneous alerts
4. **Handle the label-free nature of the problem** - no collision ground truth was provided and none could be assumed

The challenge deliberately avoided specifying the method. Teams were free to use rule-based systems, machine learning, physics-based simulation, or any combination. The KAU AE Team chose a **self-supervised deep learning** approach as the most principled way to extract learned signal from the CDM sequences without requiring labels.

A key constraint was operational realism: the output had to be something a real satellite operator could act on during a 24-hour operations cycle - not a research curiosity that required days of computation or a PhD to interpret. This is why our system produces two numbers (threat score, confidence) and four clear operator actions (ACT NOW, WATCH CLOSELY, SAFELY IGNORE, NOT PRIORITY).

---

### 2.3 ALDORIA & The CDM Dataset

**ALDORIA** (formerly known as SpacAble) is a French company specializing in Space Situational Awareness (SSA) services. They operate a network of optical sensors and maintain an independent catalogue of orbital objects, providing conjunction screening services to commercial and governmental satellite operators. Their data is notable for covering a broad population of objects - not just those tracked by the US SSN - using their own observation assets and data fusion pipeline.

For SDC2026, ALDORIA provided a proprietary dataset of **real CDMs** spanning a full calendar year of conjunction screenings. This dataset was made available to competition teams exclusively for this challenge.

Key characteristics of the ALDORIA CDM dataset:
- **Real operational data** - not simulated, not synthetic. These are actual conjunction alerts generated in the course of real satellite operations.
- **Global coverage** - screens conjunctions across the full LEO population, not just high-value payloads
- **Full CDM content** - each file contains the complete CDM body including state vectors, covariance matrices, probability calculations, and object metadata
- **Multiple CDMs per event** - the dataset captures the temporal evolution of conjunctions, not just single-point snapshots
- **Diverse tracking quality** - some objects are tracked by high-precision assets; others rely on lower-quality data, resulting in a natural range of covariance sizes

The raw data is in KVN format and cannot be redistributed due to ALDORIA's licensing terms. The `parsed_cdm_data.csv` file (213 MB) is the processed form and is included in the repository; the original KVN files are not.

---

### 2.4 What Is a CDM? (Conjunction Data Message)

A **Conjunction Data Message (CDM)** is the standard file format for communicating conjunction risk between space surveillance providers and satellite operators. It is defined by the **CCSDS (Consultative Committee for Space Data Systems) standard 508.0-B-1**, widely adopted by ESA, NASA, JAXA, and commercial operators.

A single CDM represents the state of knowledge about a conjunction **at the moment the message was created**. It contains:

**Header Section (Event-Level):**
- `CREATION_DATE` - when this CDM was generated
- `TCA` - predicted Time of Closest Approach
- `MISS_DISTANCE` - predicted closest separation at TCA (meters)
- `COLLISION_PROBABILITY` - probability of collision (dimensionless, 0 to 1)
- `RELATIVE_SPEED` - relative velocity of the two objects at TCA (m/s)
- `RELATIVE_POSITION_R/T/N` - separation in radial, transverse, normal components (meters)

**Per-Object Section (repeated for OBJECT1 and OBJECT2):**
- `OBJECT_NAME`, `OBJECT_DESIGNATOR` - identification
- `OBJECT_TYPE` - PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN
- `X`, `Y`, `Z`, `X_DOT`, `Y_DOT`, `Z_DOT` - state vector in ECI frame (km, km/s)
- `CR_R`, `CT_T`, `CN_N` - diagonal covariance matrix elements in RTN frame (m²)
  - `CR_R`: radial position uncertainty squared
  - `CT_T`: transverse position uncertainty squared
  - `CN_N`: normal (cross-track) position uncertainty squared
- Off-diagonal covariance terms (`CT_R`, `CN_R`, `CN_T`, etc.)
- `COVARIANCE_METHOD` - how the covariance was computed (e.g., CALCULATED, DEFAULT)
- `MANEUVERABLE` - YES/NO - whether the object can perform avoidance maneuvers

The **combined covariance** (summing both objects' uncertainty) determines the effective position uncertainty of the conjunction. A large combined covariance means the Pc estimate is unreliable - the actual geometry could be quite different from what's predicted.

---

### 2.5 What Is KVN Format? (Keyhole Notation)

**KVN (Keyhole Notation)** is a plain-text, line-by-line key-value format used to encode CCSDS data messages. Each line is either:
- A `KEY = VALUE [units]` pair
- A section delimiter (e.g., `OBJECT = OBJECT1`)
- A comment line starting with `COMMENT`
- A blank line

Example KVN snippet from a real CDM:
```
CCSDS_CDM_VERS          = 1.0
CREATION_DATE           = 2025-11-01T12:00:00.000
TCA                     = 2025-11-03T10:00:00.000
MISS_DISTANCE           = 150.5 [m]
COLLISION_PROBABILITY   = 1.5E-05
RELATIVE_SPEED          = 12500.0 [m/s]
OBJECT                  = OBJECT1
OBJECT_NAME             = STARLINK-1234
CR_R                    = 25.0 [m**2]
CT_T                    = 100.0 [m**2]
...
OBJECT                  = OBJECT2
OBJECT_NAME             = COSMOS-2251 DEB
CR_R                    = 8500.0 [m**2]
CT_T                    = 430000.0 [m**2]
```

The filename encodes the event identity: `CDM_<NORAD_ID_1>_<NORAD_ID_2>_<MESSAGE_ID>.kvn`. For example, `CDM_25544_48274_003.kvn` is the third CDM (message 003) for the conjunction between object 25544 and object 48274. The event_id `25544_48274` groups all CDMs for this conjunction across all message IDs.

Our parser (`step1_parse_kvn.py`) reads KVN files using regex line splitting, tracks which `OBJECT` section each field belongs to, strips units from values, and derives the `event_id` from the filename structure.

---

### 2.6 Dataset Statistics at a Glance

After parsing all KVN files through `step1_parse_kvn.py`, the resulting `parsed_cdm_data.csv` contains:

| Metric | Value |
|--------|-------|
| Total CDMs parsed | **185,511** (185,415 retained post-timestamp-repair) |
| Unique conjunction events | **64,109** (after filtering for ≥2 CDMs) |
| Unique object pairs | 64,109 distinct NORAD ID pairs |
| CDMs per event (range) | 2 - 48 |
| CDMs per event (median) | ~2 |
| Dataset time span | Jan 1 to June 30, 2024 |
| Total file size (CSV) | 213 MB |

**Split breakdown (after filtering for ≥2 CDMs per event):**

| Split | Events | Samples (self-supervised) |
|-------|--------|--------------------------|
| Training | 51,287 (80%) | 77,989 |
| Validation | 6,411 (10%) | 9,598 |
| Test | 6,411 (10%) | 9,637 |

**CDM distribution by sequence length:**
- 1 CDM: excluded (cannot form input→target pair)
- 2-3 CDMs: early-stage conjunctions, data-sparse
- 4-10 CDMs: typical operational range
- 11-20 CDMs: well-observed events
- 20+ CDMs: truncated to 20 timesteps (max_sequence_length in config.yaml)

**Collision probability distribution:**
- Minimum: effectively 0 (below floating-point precision)
- Median: ~2×10⁻⁶ (log₁₀ ≈ −5.7)
- Maximum: ~0.07 (log₁₀ ≈ −1.15) - extremely high risk event
- Distribution: heavily right-skewed; the vast majority of events have Pc well below 1×10⁻⁵

**Covariance scale (before log1p transform):**
- `combined_ct_t` (transverse): mean ~8.2 billion m², std ~19.8 billion m²
- `combined_cr_r` (radial): mean ~10.3 million m²
- `combined_cn_n` (normal): mean ~102,000 m²
- These enormous ranges are why log1p transformation was mandatory before scaling (see Section 6.4)

**Padding analysis (after sequence preparation):**
- In X_train: 69.3% of all values are the padding sentinel (-999.0)
- This reflects that most events have far fewer than 20 CDMs, leaving most timestep slots padded

---

### 2.7 What Was NOT Provided (Labels, Collision Ground Truth)

This is important to state explicitly: **the ALDORIA dataset contains no collision labels.** There is no `did_collide = True/False` column. There is no ground truth indicating which conjunction events resolved in near-miss and which resolved safely. There is no way to verify, post-hoc, whether any event resulted in a collision.

This is the standard situation in conjunction screening. Actual collisions are so rare that even a full year of global CDM data contains none with known outcome. The dataset is a record of alerts - not outcomes.

**What this means for our approach:**
- Any supervised classifier is impossible - there is nothing to classify against
- Even unsupervised anomaly detection faces the challenge that "anomalous" CDMs may simply reflect tracking peculiarities, not physical risk
- The only honest approach is to learn from the **internal structure** of the CDM sequences themselves - which is exactly what our self-supervised model does

**What was provided beyond the CDMs:**
- The raw KVN files (not redistributable)
- Competition documentation from SSA explaining the problem context
- No pre-computed features, no baseline models, no example outputs - teams started from raw files

This deliberate minimalism forced teams to think carefully about what information is actually present in CDM data and how to extract it - a design choice by the SSA that pushed toward genuinely novel approaches rather than feature engineering on pre-processed tables.

---

## 3. Literature Review & Prior Art

### 3.1 Traditional Conjunction Screening Methods

Conjunction screening has been practiced since the early days of operational spaceflight, but it matured significantly after the 2009 Iridium-Cosmos collision demonstrated the catastrophic consequences of insufficient monitoring. The standard operational workflow, still used today by most major space agencies, follows a deterministic pipeline:

1. **Catalog maintenance** - ground-based sensors (radar, optical) track objects and update their orbital elements, stored as Two-Line Elements (TLEs) or higher-fidelity state vectors
2. **Propagation** - numerical or analytical orbit propagators (SGP4, HPOP) forward-propagate states to a future epoch to predict positions
3. **Screening** - all pairs of objects are checked for close approaches within a defined screening volume (typically a box of several km × km × km around each satellite)
4. **CDM generation** - conjunction events within the screening threshold trigger CDM production by the data provider
5. **Operator notification** - operators receive CDMs and apply their own threshold rules

The primary operational systems today include:
- **US Space Surveillance Network (SSN)** / 18th Space Control Squadron - provides conjunction screening for registered operators via the space-track.org portal
- **ESA Space Debris Office** - operates the ESA conjunction screening service for ESA missions using the SOCIT4 tool
- **LeoLabs, ARES, ALDORIA** - commercial SSA providers offering independent conjunction screening with their own sensor networks

All of these systems ultimately produce CDMs and apply Pc thresholds. None learns from CDM history. None produces a confidence estimate. None distinguishes between data quality and physical risk. They are alert generators, not alert triage systems.

---

### 3.2 The ESA/NASA Maneuver Decision Threshold

The most widely cited operational threshold is **Pc > 1×10⁻⁴** (1-in-10,000), adopted by ESA as the threshold at which maneuver evaluation becomes mandatory and used by NASA as a high-concern level. The threshold's origin is partly empirical and partly economic - it represents a point where the expected value of collision damage (probability × spacecraft cost) exceeds the cost of executing a maneuver.

However, this threshold has significant known weaknesses:

**It ignores uncertainty in the Pc estimate itself.** A Pc of 1.5×10⁻⁴ calculated from a covariance that spans 100 km in the transverse direction is fundamentally different from the same Pc from a covariance that spans 10 meters. The threshold treats them identically.

**It creates discontinuous incentives.** An event at Pc = 9.9×10⁻⁵ receives no attention; one at Pc = 1.1×10⁻⁴ triggers urgent review. The underlying physics is continuous; the threshold is not.

**It does not account for trajectory evolution.** A Pc that has risen from 1×10⁻⁷ to 1×10⁻⁴ over 48 hours is far more alarming than one that has dropped from 1×10⁻² to 1×10⁻⁴ over the same period. Both trigger the threshold; neither the rate nor direction of change is captured.

Our work directly addresses this third weakness by modeling Pc trajectory - where Pc is going, not just where it is now.

---

### 3.3 The Iridium-Cosmos 2009 Event & FengYun-1C

**Iridium 33 / Cosmos 2251 - February 10, 2009**

At 16:56 UTC on February 10, 2009, Iridium 33 (a commercial communications satellite, ~560 kg) and Cosmos 2251 (a defunct Russian military communications satellite, ~950 kg) collided at approximately 789 km altitude over Siberia at a relative velocity of ~11.7 km/s. This was the first accidental hypervelocity collision between two intact satellites in history.

The collision generated an estimated 2,300+ trackable debris fragments (>10 cm), with hundreds of thousands of smaller untrackable pieces. The debris clouds spread across a broad altitude band and continue to generate conjunction alerts more than 15 years later.

What makes this event particularly relevant to our work: **the collision occurred despite the existence of conjunction screening systems**. Post-event analysis revealed that the conjunction had been screened and a CDM had been generated - but the Pc estimate was considered low enough (or the event was deprioritized) that no maneuver was executed. The Iridium operator later stated they were not notified of the conjunction at all.

This is precisely the failure mode our system is designed to prevent: a high-threat event being buried under a flood of lower-priority alerts without adequate confidence signal.

**FengYun-1C - January 11, 2007**

The FengYun-1C event was not an accidental collision but a deliberate Chinese anti-satellite test (ASAT). A ground-launched missile destroyed the defunct Chinese weather satellite FengYun-1C at ~865 km altitude, creating the largest single debris-generating event in the history of spaceflight: over 3,500 tracked fragments, many in orbits that will persist for centuries.

Both events underline that the consequences of conjunction screening failures are measured not in single satellite losses but in decade-long pollution of entire orbital shells.

---

### 3.4 Why Supervised Classification Fails Here

Multiple research groups have attempted to frame collision risk assessment as a supervised classification problem - predicting whether a given conjunction event will result in a collision. This approach fails for a fundamental reason that cannot be overcome with better algorithms: **the training data does not contain the signal needed to learn the task**.

The arguments against supervised classification:

**Class imbalance is insurmountable.** In a dataset of 185,511 CDMs spanning one year, zero resulted in confirmed collisions. Technically, the positive class does not exist in the training data. Any supervised classifier will simply learn to output "no collision" with 100% recall on the negative class and undefined performance on the positive class.

**Label noise is structural.** Even if historical collision events existed, they would represent collisions that *happened despite* screening systems - a potentially biased sample of the highest-risk, most-missed events.

**The learning target is not what operators need.** Operators do not need to know whether a collision *will* happen (impossible to know with certainty). They need to know whether a conjunction *deserves urgent attention right now, given current information quality*. These are fundamentally different questions.

**The temporal structure of CDM sequences is ignored.** A single-CDM classifier throws away all information about how the event evolved to reach this point - which is arguably the most important signal available.

Our self-supervised approach sidesteps all of these problems by reframing the learning task: instead of predicting collision outcomes, we predict CDM evolution. The collision dynamics are implicitly encoded in the sequence patterns the model learns, without ever requiring outcome labels.

---

### 3.5 Prior ML Work on Collision Risk Assessment

Despite the fundamental difficulties, several research groups have attempted ML approaches to aspects of this problem:

**PC estimation improvement:** Work by Greco et al. (2021) and others explored using Monte Carlo simulation and surrogate models to improve Pc computation speed without sacrificing accuracy. These methods improve the *quality* of individual Pc estimates but do not address the triage problem.

**Anomaly detection approaches:** Several groups have explored treating high-Pc events as anomalies and using isolation forests or autoencoders to detect them. The challenge is that anomalous CDMs often reflect sensor noise or tracking errors rather than genuine physical threats - high-Pc from a poorly-tracked object is common and benign.

**Feature-based classifiers:** Approaches using XGBoost or random forests on per-CDM features (Pc, miss distance, time to TCA) have been published. These treat each CDM independently, ignore temporal sequence, and cannot produce confidence estimates.

**Reinforcement learning for maneuver timing:** Some groups have explored RL to optimize maneuver decisions given a series of CDMs. These require a simulation environment and are not directly applicable to real CDM data without ground truth outcomes.

**CARA (Conjunction Assessment Risk Analysis):** NASA's operational system for human spaceflight uses enhanced Pc computation methods but still ultimately applies threshold-based rules to triage events.

None of these prior approaches combines: (1) temporal sequence modeling, (2) uncertainty quantification, (3) operationally actionable output, and (4) label-free training. Our system does all four.

---

### 3.6 Self-Supervised Learning in Temporal Domains

**Self-supervised learning (SSL)** is a paradigm in which a model is trained to predict some aspect of its own input - creating a pretext task that generates free supervision signal without requiring human-labeled data. It has become one of the most productive areas of deep learning:

- **BERT (2018):** Trained to predict masked words in text sequences, learning rich language representations
- **GPT family:** Trained to predict the next token in a sequence - exactly analogous to our CDM prediction formulation
- **SimCLR, MoCo (2020):** Contrastive SSL for image representations
- **Wav2Vec (2020):** SSL for speech, predicting future audio representations from past

The **next-step prediction** formulation - given a sequence of past observations, predict the next one - is among the oldest and most reliable SSL approaches. It is the basis of language models (predict next word), time series forecasting, and autoregressive generative models.

Our application to CDM sequences follows this exact formulation:
- **Input:** CDM₁, CDM₂, ..., CDMₙ₋₁ (historical conjunction measurements)
- **Target:** CDMₙ (the next CDM in the sequence)
- **Pretext task:** minimize prediction error on held-out CDMs

The key insight is that **a model cannot accurately predict CDMₙ without learning the underlying dynamics that govern conjunction evolution**. The model must implicitly learn: how Pc changes as TCA approaches, how covariance shrinks as more tracking data arrives, how miss distance evolves with trajectory refinements. These are exactly the dynamics we want to capture for risk assessment.

---

### 3.7 Bayesian Deep Learning & Uncertainty Quantification

Standard neural networks produce point estimates - a single prediction with no measure of confidence. For safety-critical applications like space operations, this is insufficient. An operator needs to know not just *what* the model predicts but *how confident* the model is in that prediction.

**Bayesian deep learning** provides a principled framework for neural network uncertainty quantification. A truly Bayesian neural network maintains a probability distribution over its weights, producing a distribution over predictions rather than a point estimate. The mean of this distribution is the best prediction; the variance captures uncertainty.

Full Bayesian inference over neural network weights is computationally intractable for modern deep networks. Several practical approximations exist:

**Monte Carlo Dropout (Gal & Ghahramani, 2016):** The most widely used approximation. Dropout - originally designed as a regularization technique - is kept active during inference. Each forward pass with different dropout masks samples a different effective neural network. Multiple passes approximate sampling from the Bayesian posterior. The variance across passes is the uncertainty estimate.

**Deep Ensembles (Lakshminarayanan et al., 2017):** Train multiple models with different random seeds, take the variance across model predictions as uncertainty. More reliable than MC Dropout but requires training N complete models.

**Variational inference:** Explicitly learns weight distributions. More principled but significantly more complex to implement.

We chose **MC Dropout** because:
1. It requires no architectural changes beyond what we already use for regularization
2. It is computationally cheap - one model, multiple forward passes
3. It has been shown to provide well-calibrated uncertainty estimates for moderate-depth networks
4. It is straightforwardly deployed: set `training=True` at inference time

The critical constraint for MC Dropout is that **BatchNormalization cannot be used** - it maintains running statistics that are fixed at inference time, breaking the "different model each pass" property. Our use of **LayerNormalization** was specifically required to make MC Dropout work correctly (see Section 7.9 and Problem 4 in Section 16).

---

### 3.8 Recurrent Neural Networks for Sequential Space Data

**Recurrent Neural Networks (RNNs)** are the natural choice for modeling sequences of varying length where order matters. In an RNN, the hidden state at each timestep is a function of the current input and the previous hidden state - allowing information to flow across the sequence.

**Long Short-Term Memory (LSTM)** networks (Hochreiter & Schmidhuber, 1997) introduced gating mechanisms (input gate, forget gate, output gate) that allow the network to selectively remember or forget information over long sequences. LSTMs dominated sequence modeling for nearly a decade.

**Gated Recurrent Units (GRU)** (Cho et al., 2014) simplify the LSTM architecture to two gates (update gate, reset gate) while achieving comparable performance on most tasks. GRUs have fewer parameters and converge faster, which is advantageous for shorter sequences like ours (2-20 CDMs).

**Bidirectional RNNs** (Schuster & Paliwal, 1997) process sequences in both forward and backward directions, concatenating the hidden states. For our pretext task (predicting CDM k from CDMs 1 to k-1), the backward direction captures information about what sequences "typically look like" from the perspective of later timesteps - adding context that improves prediction quality.

Prior applications of RNNs to space domain data include:
- Orbit prediction and conjunction screening (research prototypes, not deployed)
- Maneuver detection from TLE time series
- Spacecraft anomaly detection from telemetry streams
- Atmospheric density prediction for drag modeling

None of these prior applications combines RNNs with self-supervised CDM sequence modeling and MC Dropout uncertainty quantification for operational risk triage.

---

### 3.9 Gap in the Literature That We Fill

Synthesizing the above, the gap in the literature is clear:

| Requirement | Traditional Systems | Prior ML Work | **This Work** |
|------------|--------------------|--------------:|---------------|
| Temporal sequence modeling | ✗ | ✗ mostly | ✓ |
| No collision labels required | ✓ (rule-based) | ✗ most require labels | ✓ |
| Uncertainty quantification | ✗ | ✗ | ✓ (MC Dropout) |
| Operationally actionable output | ✓ (simple) | ✗ | ✓ (4-quadrant) |
| Distinguishes data quality from physical risk | ✗ | ✗ | ✓ |
| Learns from CDM trajectory dynamics | ✗ | ✗ | ✓ |

Our work fills all six gaps simultaneously. It is, to our knowledge, the first system to combine self-supervised temporal sequence modeling of CDM sequences with Bayesian uncertainty quantification and an operationally-structured four-quadrant output for satellite collision risk triage.
`;
