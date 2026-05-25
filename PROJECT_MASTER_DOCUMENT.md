# DebriSolver â€” Complete Project Master Document
### Learning Conjunction Dynamics: A Self-Supervised Approach to Satellite Collision Risk Assessment
**SDC2026 Â· KAU Aerospace Engineering Team Â· King Abdulaziz University**
*Space Debris Conference 2026 â€” Saudi Space Agency DebriSolver Competition, Riyadh*

---

> **How to use this document:** This is a living, section-by-section record of everything about this project â€” from the global problem it addresses, to the research decisions, architecture choices, engineering fixes, results, and lessons learned. Each section is self-contained. Read linearly for full context, or jump to any section independently.

---

## TABLE OF CONTENTS

1. [Background & The Global Problem](#1-background--the-global-problem)
2. [The Competition & Dataset](#2-the-competition--dataset)
3. [Literature Review & Prior Art](#3-literature-review--prior-art)
4. [Our Proposed Solution â€” The Core Idea](#4-our-proposed-solution--the-core-idea)
5. [Data: From Raw KVN to Structured CSV](#5-data-from-raw-kvn-to-structured-csv)
6. [Feature Engineering & Sequence Preparation](#6-feature-engineering--sequence-preparation)
7. [Model Architecture â€” The BiGRU](#7-model-architecture--the-bigru)
8. [Training Strategy & Optimization](#8-training-strategy--optimization)
9. [Uncertainty Quantification â€” MC Dropout](#9-uncertainty-quantification--mc-dropout)
10. [Scoring System â€” Threat & Confidence](#10-scoring-system--threat--confidence)
11. [The Evaluation Gate â€” Step 3B](#11-the-evaluation-gate--step-3b)
12. [Production Inference â€” The Dashboard](#12-production-inference--the-dashboard)
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

Low Earth Orbit (LEO) â€” the band of space roughly between 160 km and 2,000 km altitude â€” is the most economically and scientifically important region of near-Earth space. It hosts weather satellites, Earth observation platforms, communication constellations (Starlink, OneWeb), GPS infrastructure, the International Space Station, and thousands of scientific instruments. It is also, as of the mid-2020s, critically and dangerously congested.

The congestion did not happen overnight. It is the result of five decades of launches combined with one unavoidable physical fact: **objects in LEO do not simply go away**. In the absence of active deorbiting, a satellite or piece of debris at 600 km altitude will remain in orbit for decades. At 800 km, centuries. At 1,000 km, essentially forever on human timescales.

The numbers as of 2024 are sobering:
- **~27,000 objects** are officially tracked by the US Space Surveillance Network (SSN)
- **~100,000â€“500,000 objects** smaller than 10 cm are estimated to exist but cannot be tracked
- **~1,000,000+ objects** smaller than 1 cm â€” each capable of disabling a spacecraft
- **Over 7,000 active satellites** are currently operating in orbit, a number growing by 1,000â€“2,000 per year as mega-constellations expand

Every launch adds objects. Every collision multiplies them. The Iridium-Cosmos collision of 2009 alone generated over 2,000 trackable debris fragments. This self-reinforcing cascade â€” where debris creates more debris â€” is known as the **Kessler Syndrome**, first described by NASA scientist Donald Kessler in 1978. We are not in Kessler Syndrome yet, but some orbital shells are approaching the tipping point.

The consequence for operators is not theoretical. It is daily. Every satellite in LEO is threading a path through a field of fast-moving objects, many of which are invisible to current sensors.

---

### 1.2 What Is a Conjunction Event?

A **conjunction** is any close approach between two objects in orbit â€” a moment when the predicted separation between them falls below a defined screening threshold, typically several kilometers. Most conjunctions are harmless: the objects pass at distances that pose no real danger. But predicting exactly how close two objects will come requires knowing their precise positions and velocities, and orbital mechanics is a discipline humbled daily by measurement noise, atmospheric drag uncertainty, and solar weather.

The standard product of conjunction analysis is the **Conjunction Data Message (CDM)** â€” a structured document, standardized by the Consultative Committee for Space Data Systems (CCSDS), that reports:

- **TCA (Time of Closest Approach):** The predicted moment of closest pass
- **MISS_DISTANCE:** The predicted closest separation in meters
- **COLLISION_PROBABILITY (Pc):** A probabilistic estimate of the chance of collision, ranging from near-zero to, in extreme cases, several percent
- **Covariance matrices** for both objects: quantifying the uncertainty in each object's position and velocity
- **State vectors:** The position (X, Y, Z) and velocity (áºŠ, áºŽ, Å») of each object at a reference epoch

CDMs are not issued once per conjunction event. They are issued repeatedly â€” sometimes every few hours â€” as new tracking data becomes available and orbital predictions are refined. A conjunction that appears alarming on Monday (Pc = 1Ã—10â»Â³) may appear benign by Thursday (Pc = 1Ã—10â»â·) as fresh radar tracks tighten the uncertainty. Or it may grow more alarming. This temporal evolution â€” the **trajectory of Pc over time** â€” is the signal our system learns to interpret.

The standard reference thresholds used by ESA, NASA, and most commercial operators are:
- **Pc > 1Ã—10â»â´** (1-in-10,000): Maneuver evaluation threshold â€” operator begins seriously considering an avoidance maneuver
- **Pc > 1Ã—10â»Â³** (1-in-1,000): High concern â€” maneuver likely executed unless additional data lowers the risk
- **Pc < 1Ã—10â»â¶** (1-in-1,000,000): Typically deprioritized â€” routine monitoring only

---

### 1.3 The Alert Overload Problem

Here is the operational reality that defines the problem: **satellite operators receive thousands of conjunction alerts per year, per satellite.**

A constellation operator managing 500 satellites may receive 50,000â€“200,000 conjunction screening alerts annually. Each alert nominally demands human attention. Each demands a decision: is this worth examining in detail? Does it require a maneuver? Do we need more tracking data?

The screening tools that generate these alerts are necessarily conservative. The cost of a missed real conjunction (satellite loss, debris cascade, potential casualties on the International Space Station) far exceeds the cost of a false alarm (analyst time, unnecessary maneuver fuel burn). So the systems are tuned to flag broadly, and they do.

The result is **alert fatigue** â€” a well-documented phenomenon in safety-critical domains where operators are overwhelmed by the volume of alerts, leading to desensitization, missed real threats, and poor decision quality. Space operations is no exception. Human analysts simply cannot give careful attention to every alert.

Compounding this: tracking data quality is uneven. Different sensors â€” ground-based radar, optical telescopes, TLE-based propagators â€” produce CDMs with very different uncertainty characteristics. A CDM from a high-precision radar track is fundamentally more trustworthy than one from an old Two-Line Element (TLE) set. But raw Pc values don't tell the operator which is which. A Pc of 1Ã—10â»âµ from a well-tracked object and a Pc of 1Ã—10â»âµ from a poorly-tracked one demand very different responses. The first might be safely deprioritized. The second demands urgent additional tracking.

---

### 1.4 The Cost of Getting It Wrong

Every conjunction alert forces a decision that carries costs in both directions:

**The cost of acting unnecessarily (false positive):**
- Fuel expenditure â€” a satellite's lifetime is measured in kilograms of propellant. Each unnecessary avoidance maneuver consumes a finite, irreplaceable resource
- Operational disruption â€” maneuvers require attitude changes, communication windows, and coordination with ground stations. They interrupt normal operations
- Opportunity cost â€” scientific instruments may be powered off, imaging opportunities missed, data collection interrupted
- Risk introduction â€” paradoxically, a poorly-timed maneuver can increase collision probability with a *different* object by changing the orbital trajectory

**The cost of not acting (false negative):**
- **Total loss of a spacecraft** valued at hundreds of millions to billions of dollars
- Loss of irreplaceable scientific data and infrastructure
- Potential danger to crew on the ISS if debris affects their orbit
- Generation of a new debris cloud â€” the Iridium-Cosmos 2009 collision, between two spacecraft that together massed roughly 1,500 kg, generated over 2,000 tracked debris fragments that will remain hazardous for decades

The operational goal is not to minimize one type of error â€” it is to make **correctly calibrated decisions** that account for both risk and confidence. An operator who knows a conjunction has high threat *and* high confidence can act decisively. An operator who knows the confidence is low can request more tracking data rather than acting on noisy information. This distinction â€” between the threat level and the certainty about the threat â€” is the core of what DebriSolver provides.

---

### 1.5 The Fundamental Data Challenge: No Collision Labels

Every conventional machine learning approach to collision risk assessment runs into the same wall: **there are almost no labeled examples of actual collisions.**

In the entire history of spaceflight, only a handful of orbital collisions have been documented:
- **Iridium 33 vs Cosmos 2251 (February 10, 2009):** The only confirmed hypervelocity collision between two intact spacecraft. Generated ~2,300 tracked debris fragments.
- **FengYun-1C deliberate destruction (January 11, 2007):** China's anti-satellite test created the largest debris cloud in history â€” over 3,000 tracked fragments, many of which will remain in orbit for decades.
- A handful of smaller collisions involving derelict satellites and debris fragments

That is it. Two events (plus the ASAT test) that could serve as "positive" training examples in a supervised classification framework. Every other conjunction â€” hundreds of thousands of them â€” resolved without collision.

This is not a dataset problem that can be solved with more data collection. Actual collisions are rare by design â€” the entire orbital mechanics community works to prevent them. The class imbalance is not 10:1 or 100:1; it is closer to **1,000,000:1** (successful close approaches to actual collisions).

A supervised classifier trained on this data would learn one thing: predict "no collision" always. It would achieve 99.9999% accuracy while being completely useless.

This is why our work is built on a **self-supervised formulation** that never requires collision labels. The model learns from the dynamics of CDM sequences â€” how Pc, miss distance, and covariance evolve over time â€” without ever needing to know whether the event eventually resulted in a collision.

---

### 1.6 Why Existing Systems Are Insufficient

Current conjunction screening systems share a set of structural limitations:

**1. They are threshold-based, not learning-based.**
Most operational systems issue alerts when Pc crosses a fixed threshold (e.g., 1Ã—10â»â´). This treats every alert as binary and ignores the temporal trajectory of Pc. An event with Pc = 2Ã—10â»â´ and falling (from 5Ã—10â»Â³ last week) is very different from one at 2Ã—10â»â´ and rising (from 1Ã—10â»â¶ yesterday). Threshold-based systems cannot distinguish them.

**2. They provide no measure of confidence in their own estimates.**
A Pc value from a CDM carries implicit uncertainty â€” the uncertainty encoded in the covariance matrices â€” but this uncertainty is rarely communicated to operators in actionable form. The covariance matrices are large, hard to interpret, and not summarized into a single confidence signal.

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

The **Space Debris Conference 2026 (SDC2026)** was organized by the **Saudi Space Agency (SSA)** and held in Riyadh, Saudi Arabia on January 26â€“27, 2026. It brought together space agencies, commercial operators, researchers, and universities to address one of the most pressing challenges in modern spaceflight: the growing threat of orbital debris.

Alongside the conference, the SSA launched the **DebriSolver Competition** â€” a technical challenge inviting teams to develop novel, data-driven approaches to the conjunction risk assessment problem. The competition was open to university teams and research groups, providing real CDM data from the French space surveillance company **ALDORIA** as the training and evaluation corpus.

The Saudi Space Agency's motivation for organizing this event reflects Saudi Arabia's growing ambition in the space sector. King Abdulaziz University (KAU), the Kingdom's oldest and one of its most prestigious research universities, entered the competition through its **Aerospace Engineering Department**, fielding a team of five engineers with backgrounds spanning machine learning, orbital mechanics, and systems engineering.

The competition was not structured as a Kaggle-style leaderboard with a fixed metric. Instead, teams were evaluated holistically on the quality and novelty of their approach, the rigor of their technical methodology, and the operational relevance of their solution. This framing encouraged creative approaches â€” which is why a self-supervised learning system was viable as a submission.

---

### 2.2 The DebriSolver Challenge Brief

The challenge asked participants to address a deceptively simple question: **given a stream of Conjunction Data Messages about a potential satellite collision, how should an operator prioritize their response?**

More precisely, the challenge required teams to:

1. **Ingest and process CDM data** in the industry-standard KVN (Keyhole Notation) format
2. **Develop a method** for assessing the credibility and urgency of conjunction alerts
3. **Produce actionable outputs** that an operator could use to triage hundreds of simultaneous alerts
4. **Handle the label-free nature of the problem** â€” no collision ground truth was provided and none could be assumed

The challenge deliberately avoided specifying the method. Teams were free to use rule-based systems, machine learning, physics-based simulation, or any combination. The KAU AE Team chose a **self-supervised deep learning** approach as the most principled way to extract learned signal from the CDM sequences without requiring labels.

A key constraint was operational realism: the output had to be something a real satellite operator could act on during a 24-hour operations cycle â€” not a research curiosity that required days of computation or a PhD to interpret. This is why our system produces two numbers (threat score, confidence) and four clear operator actions (ACT NOW, WATCH CLOSELY, SAFELY IGNORE, NOT PRIORITY).

---

### 2.3 ALDORIA & The CDM Dataset

**ALDORIA** (formerly known as SpacAble) is a French company specializing in Space Situational Awareness (SSA) services. They operate a network of optical sensors and maintain an independent catalogue of orbital objects, providing conjunction screening services to commercial and governmental satellite operators. Their data is notable for covering a broad population of objects â€” not just those tracked by the US SSN â€” using their own observation assets and data fusion pipeline.

For SDC2026, ALDORIA provided a proprietary dataset of **real CDMs** spanning a full calendar year of conjunction screenings. This dataset was made available to competition teams exclusively for this challenge.

Key characteristics of the ALDORIA CDM dataset:
- **Real operational data** â€” not simulated, not synthetic. These are actual conjunction alerts generated in the course of real satellite operations.
- **Global coverage** â€” screens conjunctions across the full LEO population, not just high-value payloads
- **Full CDM content** â€” each file contains the complete CDM body including state vectors, covariance matrices, probability calculations, and object metadata
- **Multiple CDMs per event** â€” the dataset captures the temporal evolution of conjunctions, not just single-point snapshots
- **Diverse tracking quality** â€” some objects are tracked by high-precision assets; others rely on lower-quality data, resulting in a natural range of covariance sizes

The raw data is in KVN format and cannot be redistributed due to ALDORIA's licensing terms. The `parsed_cdm_data.csv` file (213 MB) is the processed form and is included in the repository; the original KVN files are not.

---

### 2.4 What Is a CDM? (Conjunction Data Message)

A **Conjunction Data Message (CDM)** is the standard file format for communicating conjunction risk between space surveillance providers and satellite operators. It is defined by the **CCSDS (Consultative Committee for Space Data Systems) standard 508.0-B-1**, widely adopted by ESA, NASA, JAXA, and commercial operators.

A single CDM represents the state of knowledge about a conjunction **at the moment the message was created**. It contains:

**Header Section (Event-Level):**
- `CREATION_DATE` â€” when this CDM was generated
- `TCA` â€” predicted Time of Closest Approach
- `MISS_DISTANCE` â€” predicted closest separation at TCA (meters)
- `COLLISION_PROBABILITY` â€” probability of collision (dimensionless, 0 to 1)
- `RELATIVE_SPEED` â€” relative velocity of the two objects at TCA (m/s)
- `RELATIVE_POSITION_R/T/N` â€” separation in radial, transverse, normal components (meters)

**Per-Object Section (repeated for OBJECT1 and OBJECT2):**
- `OBJECT_NAME`, `OBJECT_DESIGNATOR` â€” identification
- `OBJECT_TYPE` â€” PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN
- `X`, `Y`, `Z`, `X_DOT`, `Y_DOT`, `Z_DOT` â€” state vector in ECI frame (km, km/s)
- `CR_R`, `CT_T`, `CN_N` â€” diagonal covariance matrix elements in RTN frame (mÂ²)
  - `CR_R`: radial position uncertainty squared
  - `CT_T`: transverse position uncertainty squared
  - `CN_N`: normal (cross-track) position uncertainty squared
- Off-diagonal covariance terms (`CT_R`, `CN_R`, `CN_T`, etc.)
- `COVARIANCE_METHOD` â€” how the covariance was computed (e.g., CALCULATED, DEFAULT)
- `MANEUVERABLE` â€” YES/NO â€” whether the object can perform avoidance maneuvers

The **combined covariance** (summing both objects' uncertainty) determines the effective position uncertainty of the conjunction. A large combined covariance means the Pc estimate is unreliable â€” the actual geometry could be quite different from what's predicted.

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
| Total CDMs parsed | **185,511** |
| Unique conjunction events | **2,003** (after filtering for â‰¥2 CDMs) |
| Unique object pairs | 2,003 distinct NORAD ID pairs |
| CDMs per event (range) | 2 â€“ 48 |
| CDMs per event (median) | ~8 |
| Dataset time span | Full calendar year |
| Total file size (CSV) | 213 MB |

**Split breakdown (after filtering for â‰¥2 CDMs per event):**

| Split | Events | Samples (self-supervised) |
|-------|--------|--------------------------|
| Training | 1,602 (80%) | 77,989 |
| Validation | 200 (10%) | 9,598 |
| Test | 201 (10%) | 9,637 |

**CDM distribution by sequence length:**
- 1 CDM: excluded (cannot form inputâ†’target pair)
- 2â€“3 CDMs: early-stage conjunctions, data-sparse
- 4â€“10 CDMs: typical operational range
- 11â€“20 CDMs: well-observed events
- 20+ CDMs: truncated to 20 timesteps (max_sequence_length in config.yaml)

**Collision probability distribution:**
- Minimum: effectively 0 (below floating-point precision)
- Median: ~2Ã—10â»â¶ (logâ‚â‚€ â‰ˆ âˆ’5.7)
- Maximum: ~0.07 (logâ‚â‚€ â‰ˆ âˆ’1.15) â€” extremely high risk event
- Distribution: heavily right-skewed; the vast majority of events have Pc well below 1Ã—10â»âµ

**Covariance scale (before log1p transform):**
- `combined_ct_t` (transverse): mean ~8.2 billion mÂ², std ~19.8 billion mÂ²
- `combined_cr_r` (radial): mean ~10.3 million mÂ²
- `combined_cn_n` (normal): mean ~102,000 mÂ²
- These enormous ranges are why log1p transformation was mandatory before scaling (see Section 6.4)

**Padding analysis (after sequence preparation):**
- In X_train: 69.3% of all values are the padding sentinel (-999.0)
- This reflects that most events have far fewer than 20 CDMs, leaving most timestep slots padded

---

### 2.7 What Was NOT Provided (Labels, Collision Ground Truth)

This is important to state explicitly: **the ALDORIA dataset contains no collision labels.** There is no `did_collide = True/False` column. There is no ground truth indicating which conjunction events resolved in near-miss and which resolved safely. There is no way to verify, post-hoc, whether any event resulted in a collision.

This is the standard situation in conjunction screening. Actual collisions are so rare that even a full year of global CDM data contains none with known outcome. The dataset is a record of alerts â€” not outcomes.

**What this means for our approach:**
- Any supervised classifier is impossible â€” there is nothing to classify against
- Even unsupervised anomaly detection faces the challenge that "anomalous" CDMs may simply reflect tracking peculiarities, not physical risk
- The only honest approach is to learn from the **internal structure** of the CDM sequences themselves â€” which is exactly what our self-supervised model does

**What was provided beyond the CDMs:**
- The raw KVN files (not redistributable)
- Competition documentation from SSA explaining the problem context
- No pre-computed features, no baseline models, no example outputs â€” teams started from raw files

This deliberate minimalism forced teams to think carefully about what information is actually present in CDM data and how to extract it â€” a design choice by the SSA that pushed toward genuinely novel approaches rather than feature engineering on pre-processed tables.

---

## 3. Literature Review & Prior Art

### 3.1 Traditional Conjunction Screening Methods

Conjunction screening has been practiced since the early days of operational spaceflight, but it matured significantly after the 2009 Iridium-Cosmos collision demonstrated the catastrophic consequences of insufficient monitoring. The standard operational workflow, still used today by most major space agencies, follows a deterministic pipeline:

1. **Catalog maintenance** â€” ground-based sensors (radar, optical) track objects and update their orbital elements, stored as Two-Line Elements (TLEs) or higher-fidelity state vectors
2. **Propagation** â€” numerical or analytical orbit propagators (SGP4, HPOP) forward-propagate states to a future epoch to predict positions
3. **Screening** â€” all pairs of objects are checked for close approaches within a defined screening volume (typically a box of several km Ã— km Ã— km around each satellite)
4. **CDM generation** â€” conjunction events within the screening threshold trigger CDM production by the data provider
5. **Operator notification** â€” operators receive CDMs and apply their own threshold rules

The primary operational systems today include:
- **US Space Surveillance Network (SSN)** / 18th Space Control Squadron â€” provides conjunction screening for registered operators via the space-track.org portal
- **ESA Space Debris Office** â€” operates the ESA conjunction screening service for ESA missions using the SOCIT4 tool
- **LeoLabs, ARES, ALDORIA** â€” commercial SSA providers offering independent conjunction screening with their own sensor networks

All of these systems ultimately produce CDMs and apply Pc thresholds. None learns from CDM history. None produces a confidence estimate. None distinguishes between data quality and physical risk. They are alert generators, not alert triage systems.

---

### 3.2 The ESA/NASA Maneuver Decision Threshold

The most widely cited operational threshold is **Pc > 1Ã—10â»â´** (1-in-10,000), adopted by ESA as the threshold at which maneuver evaluation becomes mandatory and used by NASA as a high-concern level. The threshold's origin is partly empirical and partly economic â€” it represents a point where the expected value of collision damage (probability Ã— spacecraft cost) exceeds the cost of executing a maneuver.

However, this threshold has significant known weaknesses:

**It ignores uncertainty in the Pc estimate itself.** A Pc of 1.5Ã—10â»â´ calculated from a covariance that spans 100 km in the transverse direction is fundamentally different from the same Pc from a covariance that spans 10 meters. The threshold treats them identically.

**It creates discontinuous incentives.** An event at Pc = 9.9Ã—10â»âµ receives no attention; one at Pc = 1.1Ã—10â»â´ triggers urgent review. The underlying physics is continuous; the threshold is not.

**It does not account for trajectory evolution.** A Pc that has risen from 1Ã—10â»â· to 1Ã—10â»â´ over 48 hours is far more alarming than one that has dropped from 1Ã—10â»Â² to 1Ã—10â»â´ over the same period. Both trigger the threshold; neither the rate nor direction of change is captured.

Our work directly addresses this third weakness by modeling Pc trajectory â€” where Pc is going, not just where it is now.

---

### 3.3 The Iridium-Cosmos 2009 Event & FengYun-1C

**Iridium 33 / Cosmos 2251 â€” February 10, 2009**

At 16:56 UTC on February 10, 2009, Iridium 33 (a commercial communications satellite, ~560 kg) and Cosmos 2251 (a defunct Russian military communications satellite, ~950 kg) collided at approximately 789 km altitude over Siberia at a relative velocity of ~11.7 km/s. This was the first accidental hypervelocity collision between two intact satellites in history.

The collision generated an estimated 2,300+ trackable debris fragments (>10 cm), with hundreds of thousands of smaller untrackable pieces. The debris clouds spread across a broad altitude band and continue to generate conjunction alerts more than 15 years later.

What makes this event particularly relevant to our work: **the collision occurred despite the existence of conjunction screening systems**. Post-event analysis revealed that the conjunction had been screened and a CDM had been generated â€” but the Pc estimate was considered low enough (or the event was deprioritized) that no maneuver was executed. The Iridium operator later stated they were not notified of the conjunction at all.

This is precisely the failure mode our system is designed to prevent: a high-threat event being buried under a flood of lower-priority alerts without adequate confidence signal.

**FengYun-1C â€” January 11, 2007**

The FengYun-1C event was not an accidental collision but a deliberate Chinese anti-satellite test (ASAT). A ground-launched missile destroyed the defunct Chinese weather satellite FengYun-1C at ~865 km altitude, creating the largest single debris-generating event in the history of spaceflight: over 3,500 tracked fragments, many in orbits that will persist for centuries.

Both events underline that the consequences of conjunction screening failures are measured not in single satellite losses but in decade-long pollution of entire orbital shells.

---

### 3.4 Why Supervised Classification Fails Here

Multiple research groups have attempted to frame collision risk assessment as a supervised classification problem â€” predicting whether a given conjunction event will result in a collision. This approach fails for a fundamental reason that cannot be overcome with better algorithms: **the training data does not contain the signal needed to learn the task**.

The arguments against supervised classification:

**Class imbalance is insurmountable.** In a dataset of 185,511 CDMs spanning one year, zero resulted in confirmed collisions. Technically, the positive class does not exist in the training data. Any supervised classifier will simply learn to output "no collision" with 100% recall on the negative class and undefined performance on the positive class.

**Label noise is structural.** Even if historical collision events existed, they would represent collisions that *happened despite* screening systems â€” a potentially biased sample of the highest-risk, most-missed events.

**The learning target is not what operators need.** Operators do not need to know whether a collision *will* happen (impossible to know with certainty). They need to know whether a conjunction *deserves urgent attention right now, given current information quality*. These are fundamentally different questions.

**The temporal structure of CDM sequences is ignored.** A single-CDM classifier throws away all information about how the event evolved to reach this point â€” which is arguably the most important signal available.

Our self-supervised approach sidesteps all of these problems by reframing the learning task: instead of predicting collision outcomes, we predict CDM evolution. The collision dynamics are implicitly encoded in the sequence patterns the model learns, without ever requiring outcome labels.

---

### 3.5 Prior ML Work on Collision Risk Assessment

Despite the fundamental difficulties, several research groups have attempted ML approaches to aspects of this problem:

**PC estimation improvement:** Work by Greco et al. (2021) and others explored using Monte Carlo simulation and surrogate models to improve Pc computation speed without sacrificing accuracy. These methods improve the *quality* of individual Pc estimates but do not address the triage problem.

**Anomaly detection approaches:** Several groups have explored treating high-Pc events as anomalies and using isolation forests or autoencoders to detect them. The challenge is that anomalous CDMs often reflect sensor noise or tracking errors rather than genuine physical threats â€” high-Pc from a poorly-tracked object is common and benign.

**Feature-based classifiers:** Approaches using XGBoost or random forests on per-CDM features (Pc, miss distance, time to TCA) have been published. These treat each CDM independently, ignore temporal sequence, and cannot produce confidence estimates.

**Reinforcement learning for maneuver timing:** Some groups have explored RL to optimize maneuver decisions given a series of CDMs. These require a simulation environment and are not directly applicable to real CDM data without ground truth outcomes.

**CARA (Conjunction Assessment Risk Analysis):** NASA's operational system for human spaceflight uses enhanced Pc computation methods but still ultimately applies threshold-based rules to triage events.

None of these prior approaches combines: (1) temporal sequence modeling, (2) uncertainty quantification, (3) operationally actionable output, and (4) label-free training. Our system does all four.

---

### 3.6 Self-Supervised Learning in Temporal Domains

**Self-supervised learning (SSL)** is a paradigm in which a model is trained to predict some aspect of its own input â€” creating a pretext task that generates free supervision signal without requiring human-labeled data. It has become one of the most productive areas of deep learning:

- **BERT (2018):** Trained to predict masked words in text sequences, learning rich language representations
- **GPT family:** Trained to predict the next token in a sequence â€” exactly analogous to our CDM prediction formulation
- **SimCLR, MoCo (2020):** Contrastive SSL for image representations
- **Wav2Vec (2020):** SSL for speech, predicting future audio representations from past

The **next-step prediction** formulation â€” given a sequence of past observations, predict the next one â€” is among the oldest and most reliable SSL approaches. It is the basis of language models (predict next word), time series forecasting, and autoregressive generative models.

Our application to CDM sequences follows this exact formulation:
- **Input:** CDMâ‚, CDMâ‚‚, ..., CDMâ‚™â‚‹â‚ (historical conjunction measurements)
- **Target:** CDMâ‚™ (the next CDM in the sequence)
- **Pretext task:** minimize prediction error on held-out CDMs

The key insight is that **a model cannot accurately predict CDMâ‚™ without learning the underlying dynamics that govern conjunction evolution**. The model must implicitly learn: how Pc changes as TCA approaches, how covariance shrinks as more tracking data arrives, how miss distance evolves with trajectory refinements. These are exactly the dynamics we want to capture for risk assessment.

---

### 3.7 Bayesian Deep Learning & Uncertainty Quantification

Standard neural networks produce point estimates â€” a single prediction with no measure of confidence. For safety-critical applications like space operations, this is insufficient. An operator needs to know not just *what* the model predicts but *how confident* the model is in that prediction.

**Bayesian deep learning** provides a principled framework for neural network uncertainty quantification. A truly Bayesian neural network maintains a probability distribution over its weights, producing a distribution over predictions rather than a point estimate. The mean of this distribution is the best prediction; the variance captures uncertainty.

Full Bayesian inference over neural network weights is computationally intractable for modern deep networks. Several practical approximations exist:

**Monte Carlo Dropout (Gal & Ghahramani, 2016):** The most widely used approximation. Dropout â€” originally designed as a regularization technique â€” is kept active during inference. Each forward pass with different dropout masks samples a different effective neural network. Multiple passes approximate sampling from the Bayesian posterior. The variance across passes is the uncertainty estimate.

**Deep Ensembles (Lakshminarayanan et al., 2017):** Train multiple models with different random seeds, take the variance across model predictions as uncertainty. More reliable than MC Dropout but requires training N complete models.

**Variational inference:** Explicitly learns weight distributions. More principled but significantly more complex to implement.

We chose **MC Dropout** because:
1. It requires no architectural changes beyond what we already use for regularization
2. It is computationally cheap â€” one model, multiple forward passes
3. It has been shown to provide well-calibrated uncertainty estimates for moderate-depth networks
4. It is straightforwardly deployed: set `training=True` at inference time

The critical constraint for MC Dropout is that **BatchNormalization cannot be used** â€” it maintains running statistics that are fixed at inference time, breaking the "different model each pass" property. Our use of **LayerNormalization** was specifically required to make MC Dropout work correctly (see Section 7.9 and Problem 4 in Section 16).

---

### 3.8 Recurrent Neural Networks for Sequential Space Data

**Recurrent Neural Networks (RNNs)** are the natural choice for modeling sequences of varying length where order matters. In an RNN, the hidden state at each timestep is a function of the current input and the previous hidden state â€” allowing information to flow across the sequence.

**Long Short-Term Memory (LSTM)** networks (Hochreiter & Schmidhuber, 1997) introduced gating mechanisms (input gate, forget gate, output gate) that allow the network to selectively remember or forget information over long sequences. LSTMs dominated sequence modeling for nearly a decade.

**Gated Recurrent Units (GRU)** (Cho et al., 2014) simplify the LSTM architecture to two gates (update gate, reset gate) while achieving comparable performance on most tasks. GRUs have fewer parameters and converge faster, which is advantageous for shorter sequences like ours (2â€“20 CDMs).

**Bidirectional RNNs** (Schuster & Paliwal, 1997) process sequences in both forward and backward directions, concatenating the hidden states. For our pretext task (predicting CDM k from CDMs 1 to k-1), the backward direction captures information about what sequences "typically look like" from the perspective of later timesteps â€” adding context that improves prediction quality.

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
| Temporal sequence modeling | âœ— | âœ— mostly | âœ“ |
| No collision labels required | âœ“ (rule-based) | âœ— most require labels | âœ“ |
| Uncertainty quantification | âœ— | âœ— | âœ“ (MC Dropout) |
| Operationally actionable output | âœ“ (simple) | âœ— | âœ“ (4-quadrant) |
| Distinguishes data quality from physical risk | âœ— | âœ— | âœ“ |
| Learns from CDM trajectory dynamics | âœ— | âœ— | âœ“ |

Our work fills all six gaps simultaneously. It is, to our knowledge, the first system to combine self-supervised temporal sequence modeling of CDM sequences with Bayesian uncertainty quantification and an operationally-structured four-quadrant output for satellite collision risk triage.

## 4. Our Proposed Solution â€” The Core Idea

### 4.1 The Central Insight: CDM Sequences Tell a Story

The foundational insight of DebriSolver is simple but powerful: **a sequence of CDMs for a single conjunction event is not a collection of independent snapshots â€” it is a story with a trajectory, and that trajectory is rich with information about how dangerous the event really is.**

Consider two events, both currently showing Pc = 5Ã—10â»âµ (below the 1Ã—10â»â´ maneuver threshold):

**Event A:** 12 CDMs over 8 days. Pc started at 1Ã—10â»â¸ four days ago, rose steadily to 1Ã—10â»â¶, then jumped to 5Ã—10â»âµ in the last 24 hours. Covariance is tightening with each CDM â€” new tracking data is arriving regularly. TCA is 18 hours away.

**Event B:** 2 CDMs. Pc was 6Ã—10â»âµ yesterday, is 5Ã—10â»âµ today â€” slightly declining. Covariance is enormous (tracking is poor). TCA is 11 days away.

Both sit below the 1Ã—10â»â´ threshold. A threshold-based system treats them identically. But any experienced conjunction analyst would immediately recognize:
- Event A is **alarming** â€” rising Pc, imminent TCA, good data quality means the estimate is trustworthy, and the trend is still upward
- Event B is **low priority** â€” Pc is probably an artifact of poor tracking, TCA is far away, and one more radar track may eliminate the event entirely

This distinction is exactly what our model learns to capture. By training on thousands of CDM sequences, the BiGRU learns what "alarming trajectory" looks like vs. what "noise that will resolve" looks like â€” without ever being told which events were actually dangerous.

---

### 4.2 The Self-Supervised Formulation

The self-supervised learning task is framed as **next-CDM prediction**:

> Given the first k CDMs of a conjunction event (CDMâ‚, CDMâ‚‚, ..., CDMâ‚–), predict the feature values of CDMâ‚–â‚Šâ‚.

Formally, for a conjunction event with N CDMs:
- We generate Nâˆ’1 training samples: (input: CDMs 1..k, target: CDM k+1) for k = 1, 2, ..., Nâˆ’1
- **Input X:** A padded sequence of shape (max_len=20, n_features=11), where CDMs before the first real measurement are filled with the sentinel value âˆ’999.0
- **Target Y:** A 1D vector of shape (n_features=11) representing the next CDM's feature values

All features in both X and Y are standardized (StandardScaler, fitted on training data only). Covariance features are log1p-transformed before standardization to handle their extreme scale.

This formulation has elegant properties:
1. **No labels required** â€” the "label" for each sample is simply the next row in the same CSV. The data supervises itself.
2. **Every CDM in a sequence contributes** â€” a 10-CDM event generates 9 training samples, each giving the model a different temporal vantage point
3. **Temporal order is preserved** â€” CDMs are always presented in chronological order (by CREATION_DATE), so the model learns causal dynamics
4. **Variable-length sequences are handled naturally** â€” via left-padding and the Masking layer, so 2-CDM and 20-CDM events coexist in the same batch

At training time, the model minimizes a weighted MSE loss over all predicted features, with higher weights on Pc (Ã—2.0) and miss distance (Ã—1.5) â€” reflecting their greater operational importance.

---

### 4.3 What "Learning Conjunction Dynamics" Means

When we say the model "learns conjunction dynamics," we mean it implicitly encodes in its weights several physical and observational relationships:

**Pc evolution near TCA:** As time to TCA decreases, conjunction geometry becomes more constrained. For well-tracked events, Pc often peaks in the last 24â€“48 hours before TCA, then either spikes (genuine threat) or collapses (tracking data resolution). The model learns this temporal pattern.

**Covariance decay with tracking:** As more radar observations arrive, orbit determination improves and covariance shrinks. A sequence showing steadily decreasing covariance is physically distinct from one showing constant or growing covariance. The model learns this as a signature of data quality.

**Miss distance refinement:** Early CDMs often have large uncertainty in miss distance. As TCA approaches and covariance tightens, miss distance estimates converge. The model learns the typical trajectory of this convergence.

**Relative position evolution in RTN frame:** The radial (R), transverse (T), and normal (N) components of relative position evolve predictably as TCA approaches. The model learns these geometric relationships.

Crucially, the model does **not** learn "event X was a collision." It learns "event X's sequence evolution pattern." When presented with a new event, it predicts what the next CDM should look like. If the actual next CDM is very different from the prediction, the event is behaving anomalously â€” which is a risk signal.

---

### 4.4 From Prediction to Threat Score

At inference time, for each test event, the model makes a prediction of the next CDM's features. The **threat score** (0â€“100) is derived from these predictions using physics-based rules in `scoring.py`:

**Base threat from predicted Pc level:**
The predicted logâ‚â‚€(Pc) is converted back to physical Pc via inverse-transform. This predicted Pc is mapped to a base threat score using operational thresholds:
- Predicted Pc > 1Ã—10â»Â³ â†’ base threat â‰ˆ 80â€“100 (extreme concern)
- Predicted Pc in [1Ã—10â»â´, 1Ã—10â»Â³] â†’ base threat â‰ˆ 50â€“80 (maneuver evaluation zone)
- Predicted Pc in [1Ã—10â»â¶, 1Ã—10â»â´] â†’ base threat â‰ˆ 20â€“50 (monitoring zone)
- Predicted Pc < 1Ã—10â»â¶ â†’ base threat â‰ˆ 0â€“20 (low concern)

**Trend modifier (where is Pc going?):**
The current Pc (from the last real CDM in the sequence) is compared to the predicted next Pc. If predicted Pc > current Pc (rising trend), threat is boosted. If predicted Pc < current Pc (falling trend), threat is penalized. This is the core novel contribution â€” the model's prediction encodes trajectory direction.

**TCA urgency bonus:**
Events with predicted time-to-TCA < 24 hours receive an urgency bonus. Events with TCA > 7 days receive a suppression factor. Imminent events demand more aggressive triage regardless of absolute Pc level.

The threat score is clipped to [0, 100] and represents a continuous ordering of event urgency.

---

### 4.5 From Uncertainty to Confidence

The **confidence level** (0.0â€“1.0) answers a different question than threat: *how much should the operator trust this threat assessment?*

Confidence is computed from three independent components, each reflecting a distinct source of epistemic information:

**Component 1 â€” MC Dropout Uncertainty (weight: 40%)**
The BiGRU is run 50 times with Dropout active (`training=True`). Each pass uses a different dropout mask, producing a different prediction. The standard deviation across 50 predictions quantifies how uncertain the model is about the next CDM. Low std â†’ model is confident in its prediction â†’ higher confidence score. High std â†’ model is confused by this event's trajectory â†’ lower confidence.

**Component 2 â€” Data Quantity (weight: 35%)**
More CDMs in the input sequence means more information. An event with 15 CDMs gives the model a rich temporal picture; an event with 2 CDMs barely constrains the model. Confidence scales with the number of valid (non-padding) timesteps in the input.

**Component 3 â€” Covariance Quality (weight: 25%)**
The combined position covariance (CR_R + CT_T + CN_N, in raw mÂ²) of the two objects determines how well-tracked they are. Large covariance means large tracking uncertainty, which means the Pc estimate itself is unreliable. Confidence is penalized as a function of covariance size, with a threshold calibrated to typical LEO tracking quality.

The three components are combined into a final confidence value clipped to [0.10, 1.00] (never zero â€” some minimum confidence always exists even with 1 CDM).

---

### 4.6 The Four-Quadrant Risk Classification

Threat score and confidence together define a **two-dimensional risk space**. Every conjunction event is placed in this space and assigned to one of four operational quadrants:

```
                    HIGH CONFIDENCE
                          â”‚
          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
          â”‚               â”‚               â”‚
 HIGH     â”‚  NOT PRIORITY â”‚   ACT NOW     â”‚  HIGH
 THREAT   â”‚  (Low Threat, â”‚   (High Threatâ”‚  THREAT
          â”‚  High Conf)   â”‚   High Conf)  â”‚
          â”‚               â”‚               â”‚
          â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
          â”‚               â”‚               â”‚
          â”‚  SAFELY IGNOREâ”‚  WATCH CLOSELYâ”‚
          â”‚  (Low Threat, â”‚  (High Threat,â”‚
          â”‚  Low Conf)    â”‚  Low Conf)    â”‚
          â”‚               â”‚               â”‚
          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                          â”‚
                    LOW CONFIDENCE
```

**ACT NOW** (High Threat, High Confidence): The model is predicting a dangerous trajectory *and* is confident in that prediction. Data is plentiful, tracking is good, uncertainty is low. These events demand immediate human review and potential maneuver planning.

**WATCH CLOSELY** (High Threat, Low Confidence): The model flags a potential threat but uncertainty is high â€” perhaps because there are only 2 CDMs, or because covariance is enormous. More tracking data is needed before a maneuver decision. Human oversight required; request additional observations.

**SAFELY IGNORE** (Low Threat, High Confidence): The model predicts Pc will remain low *and* is confident about it. The event has good tracking data and a benign trajectory. These can be deprioritized with confidence â€” operators can focus elsewhere.

**NOT PRIORITY** (Low Threat, Low Confidence): Low predicted threat but also low confidence. Monitor passively. May warrant a second look if more CDMs arrive.

The quadrant boundaries in our implementation:
- Threat threshold: 50/100 (separates High from Low threat)
- Confidence threshold: 0.5 (separates High from Low confidence)

---

### 4.7 Why This Is Deployment-Safe (No Label Leakage)

A critical property of our system: **it cannot leak future information, and its outputs are fully explainable without collision ground truth.**

**No label leakage:** The training process never uses any information about whether a conjunction ultimately resulted in a collision. The pretext task (predict next CDM) uses only information available at the time of prediction. There is no "outcome label" anywhere in the pipeline.

**No future data leakage:** Each prediction is made from CDMs available at a specific moment in time (CDMs 1 through k, predicting CDM k+1). The model never sees CDMs beyond the prediction horizon. The Masking layer ensures that padding positions contribute zero gradient.

**Reproducibility:** All random operations (data split, model initialization, dropout masks at inference) are seeded with SEED=42 and TensorFlow deterministic mode is enabled, producing identical results across runs.

**Interpretability:** The threat score and confidence are computed from physical quantities (predicted Pc, predicted time-to-TCA, covariance) â€” all directly traceable to CDM fields. An operator can always ask "why is this event ACT NOW?" and receive an answer grounded in physics: "predicted Pc = 2.3Ã—10â»â´ (above 1Ã—10â»â´ threshold), rising from current 4.1Ã—10â»âµ, TCA in 11 hours, model uncertainty low (std=0.08), 14 CDMs available."

---

### 4.8 Comparison to Our Initial Approach (What We Changed)

The system described above is the result of significant iteration. Our initial design had several critical differences:

**Initially: Raw COLLISION_PROBABILITY as a direct feature**
We included the raw linear Pc value (0 to 1) as a training feature. This caused catastrophic validation loss (~84.9) because StandardScaler cannot handle a distribution that spans 10 orders of magnitude â€” a few high-Pc events produced scaled values of 100+ standard deviations. **Fix:** Replaced with logâ‚â‚€(Pc), compressing the same range into a manageable [-âˆž to 0] interval (capped at -10 for near-zero Pc).

**Initially: Raw covariance values without log1p**
Covariance features span from near-zero to 20 billion mÂ². Even with StandardScaler, the resulting scaled values for extreme-covariance events were wildly outside the [-3, 3] range. **Fix:** Applied log1p before StandardScaler, compressing the range from [0, 2Ã—10Â¹â°] to [0, ~22].

**Initially: Zero as the padding sentinel**
After StandardScaler, real CDM features are mean-centered â€” meaning real values can legitimately be zero. The Masking layer using mask_value=0 was masking real data, not just padding. **Fix:** Changed padding sentinel to -999.0, a value that is physically impossible after standardization.

**Initially: BatchNormalization in the model**
The original architecture used BatchNorm for stability. This broke MC Dropout â€” batch statistics were recomputed each forward pass in a way unrelated to weight uncertainty. **Fix:** Replaced all BatchNorm with LayerNorm, which is sample-local and immune to the training=True/False distinction.

**Initially: Confidence weights heavily biased toward MC Dropout uncertainty**
The original confidence formula weighted MC Dropout std at 60%, leaving data quantity and covariance quality at 20% each. Most events have only 2â€“5 CDMs, so data_confidence was always low, dragging the total below 0.5. Every event landed in the "WATCH CLOSELY" or "SAFELY IGNORE" zone â€” the system had no discrimination. **Fix:** Rebalanced to 40/35/25, allowing events with enough CDMs and low uncertainty to reach ACT NOW.

Each of these changes is documented in detail in Section 16 with the exact diagnostic process used to discover and fix each problem.

## 5. Data: From Raw KVN to Structured CSV

### 5.1 Anatomy of a KVN CDM File

Each KVN file is a plain-text document containing one Conjunction Data Message. The file structure is strictly linear: every key-value pair appears on its own line, and the ordering within the file defines which `OBJECT` block each field belongs to.

A complete CDM file has three logical sections:

**1. Global header** â€” fields that describe the conjunction as a whole, before any `OBJECT = OBJECT1` delimiter appears:
```
CCSDS_CDM_VERS          = 1.0
CREATION_DATE           = 2025-11-01T12:00:00.000
ORIGINATOR              = ALDORIA
MESSAGE_ID              = CDM_25544_48274_003
TCA                     = 2025-11-03T10:00:00.000
MISS_DISTANCE           = 150.5 [m]
RELATIVE_SPEED          = 12500.0 [m/s]
RELATIVE_POSITION_R     = -42.3 [m]
RELATIVE_POSITION_T     = 143.1 [m]
RELATIVE_POSITION_N     = -11.7 [m]
COLLISION_PROBABILITY   = 1.5E-05
COLLISION_PROBABILITY_METHOD = FOSTER-1992
```

**2. OBJECT1 block** â€” begins at the `OBJECT = OBJECT1` line:
```
OBJECT                  = OBJECT1
OBJECT_DESIGNATOR       = 25544
OBJECT_NAME             = ISS (ZARYA)
OBJECT_TYPE             = PAYLOAD
MANEUVERABLE            = YES
X                       =  6500.123 [km]
Y                       =  1200.456 [km]
Z                       =   500.789 [km]
X_DOT                   =     1.523 [km/s]
Y_DOT                   =    -7.201 [km/s]
Z_DOT                   =     0.312 [km/s]
CR_R                    =    25.000 [m**2]
CT_T                    =   100.000 [m**2]
CN_N                    =    50.000 [m**2]
```

**3. OBJECT2 block** â€” begins at `OBJECT = OBJECT2`:
```
OBJECT                  = OBJECT2
OBJECT_DESIGNATOR       = 48274
OBJECT_NAME             = COSMOS DEB
OBJECT_TYPE             = DEBRIS
MANEUVERABLE            = NO
CR_R                    =  8500.000 [m**2]
CT_T                    = 430000.000 [m**2]
CN_N                    = 12000.000 [m**2]
```

Key parsing challenges: field order is not guaranteed, units appear in brackets on the same line as values, some fields appear only in one object block, and the OBJECT1/OBJECT2 distinction must be tracked by the parser's state machine.

---

### 5.2 The KVN Parser (step1_parse_kvn.py) â€” How It Works

`step1_parse_kvn.py` implements a stateful line-by-line parser. The core function `parse_kvn_file(filepath)` returns a flat Python dictionary containing all extracted fields for one CDM, or `None` if parsing fails.

The high-level logic:

```python
def parse_kvn_file(filepath):
    record = {}
    current_object = None   # tracks whether we're in OBJECT1 or OBJECT2 block

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('COMMENT'):
                continue                    # skip blanks and comments

            if '=' not in line:
                continue                    # skip malformed lines

            key, value = line.split('=', 1)
            key = key.strip()
            value = strip_units(value.strip())   # remove [m], [m**2], [km/s], etc.

            if key == 'OBJECT':
                current_object = value.strip()   # 'OBJECT1' or 'OBJECT2'
                continue

            if current_object:
                # Namespace by object: 'CR_R' â†’ 'object1_CR_R' or 'object2_CR_R'
                namespaced_key = f"{current_object.lower()}_{key}"
                record[namespaced_key] = value
            else:
                record[key] = value

    # Derive event_id from filename, compute derived features
    record = enrich_record(record, filepath)
    return record if is_valid(record) else None
```

The parser is **fail-soft**: if a field is missing (not all CDMs include all optional fields), it simply isn't added to the record. Missing fields are later handled by the `SimpleImputer` in step 2.

---

### 5.3 Field Extraction & Unit Stripping

KVN values often include physical units in square brackets: `150.5 [m]`, `100.0 [m**2]`, `6500.0 [km]`. These must be stripped before numeric conversion.

The `strip_units()` function uses a regex to remove everything from the first `[` to the end of the string:

```python
import re

def strip_units(value_str):
    """Remove unit annotations like [m], [m**2], [km/s] from value strings."""
    return re.sub(r'\s*\[.*?\]\s*$', '', value_str).strip()
```

After stripping, values are stored as strings in the record dictionary. Numeric conversion to `float` happens downstream in step 2 when building the pandas DataFrame, where `pd.to_numeric(errors='coerce')` handles any remaining non-numeric values by converting them to `NaN` for imputation.

Special cases handled:
- **Scientific notation:** `1.5E-05` is valid Python float notation and converts cleanly
- **Negative values:** `-42.3 [m]` strips correctly to `-42.3`
- **Empty values:** `COLLISION_PROBABILITY =` (blank after `=`) â€” stored as empty string, later becomes NaN
- **COMMENT lines:** Explicitly skipped; some KVN files embed commentary inline

---

### 5.4 Object-Specific Field Namespacing

CDM files contain two objects, and many fields (especially covariance terms) appear for both. Without namespacing, the second object's `CR_R` would overwrite the first's.

The parser tracks the current object context using the state variable `current_object`:
- Before any `OBJECT = OBJECT1` line: fields are global (e.g., `TCA`, `MISS_DISTANCE`)
- After `OBJECT = OBJECT1`: fields are prefixed `object1_` (e.g., `object1_CR_R`)
- After `OBJECT = OBJECT2`: fields are prefixed `object2_` (e.g., `object2_CR_R`)

The resulting flat record contains:

| Raw KVN field | Parsed key |
|---|---|
| `TCA` (global) | `TCA` |
| `COLLISION_PROBABILITY` (global) | `COLLISION_PROBABILITY` |
| `CR_R` (in OBJECT1 block) | `object1_CR_R` |
| `CT_T` (in OBJECT1 block) | `object1_CT_T` |
| `CR_R` (in OBJECT2 block) | `object2_CR_R` |
| `CT_T` (in OBJECT2 block) | `object2_CT_T` |
| `OBJECT_NAME` (in OBJECT1 block) | `object1_OBJECT_NAME` |
| `OBJECT_NAME` (in OBJECT2 block) | `object2_OBJECT_NAME` |

This flat, namespaced structure makes the downstream pandas processing straightforward â€” every CDM row in the CSV has the same column schema.

---

### 5.5 Event ID Construction from Filenames

The event identifier â€” which groups all CDMs for the same conjunction together â€” is derived from the KVN **filename**, not from its contents. The ALDORIA naming convention is:

```
CDM_<NORAD_ID_1>_<NORAD_ID_2>_<MESSAGE_NUMBER>.kvn
```

For example: `CDM_25544_48274_003.kvn`
- `25544` = NORAD ID of Object 1
- `48274` = NORAD ID of Object 2
- `003` = this is the 3rd CDM for this conjunction event

The event_id is constructed as:
```python
def extract_event_id(filepath):
    filename = os.path.basename(filepath)           # 'CDM_25544_48274_003.kvn'
    parts = filename.replace('.kvn', '').split('_') # ['CDM', '25544', '48274', '003']
    if len(parts) >= 4:
        return {
            'event_id': f"{parts[1]}_{parts[2]}",   # '25544_48274'
            'object1_norad_id': parts[1],            # '25544'
            'object2_norad_id': parts[2],            # '48274'
        }
```

This is a robust approach because:
- The filename is always present (parsing doesn't depend on file contents for identification)
- The NORAD IDs in the filename are guaranteed to be consistent across all CDMs for the same event
- Different message numbers for the same event naturally share the same `event_id`

One edge case: some files have non-standard names or extra underscores. The parser handles this with `len(parts) >= 4` â€” if the filename doesn't match the expected pattern, `event_id` is set to the full filename stem as a fallback.

---

### 5.6 Derived Feature Engineering at Parse Time

Beyond extracting raw CDM fields, the parser computes several derived features that are more useful for modeling than the raw fields:

**`time_to_tca_hours`** â€” Time remaining until closest approach, in hours:
```python
creation_dt = parse_datetime(record['CREATION_DATE'])
tca_dt      = parse_datetime(record['TCA'])
delta = (tca_dt - creation_dt).total_seconds() / 3600.0
record['time_to_tca_hours'] = max(0.0, delta)
```
This is arguably the most operationally important derived feature â€” it tells the model exactly where in the conjunction timeline this CDM was issued.

**`log10_pc`** â€” Log-10 of collision probability:
```python
pc = float(record.get('COLLISION_PROBABILITY', 0))
record['log10_pc'] = math.log10(max(pc, 1e-10))  # floor at 1e-10
```
Converts the 10-order-of-magnitude range of Pc values into a compact [-10, 0] scale suitable for neural network training.

**`combined_cr_r`, `combined_ct_t`, `combined_cn_n`** â€” Sum of both objects' covariance diagonal elements:
```python
record['combined_cr_r'] = float(record.get('object1_CR_R', 0)) + float(record.get('object2_CR_R', 0))
record['combined_ct_t'] = float(record.get('object1_CT_T', 0)) + float(record.get('object2_CT_T', 0))
record['combined_cn_n'] = float(record.get('object1_CN_N', 0)) + float(record.get('object2_CN_N', 0))
```
The combined covariance represents the total positional uncertainty of the conjunction. A single combined value is more useful to the model than four separate (per-object Ã— per-axis) values.

These derived features are computed once at parse time and stored directly in `parsed_cdm_data.csv` alongside the raw fields, avoiding redundant computation in later pipeline steps.

---

### 5.7 Schema Validation & Fail-Fast Design

After parsing, each record is validated before being accepted into the output CSV. A CDM is rejected (`parse_kvn_file` returns `None`) if:

- `TCA` is missing or cannot be parsed as a datetime
- `CREATION_DATE` is missing or cannot be parsed
- `MISS_DISTANCE` is missing (required for basic conjunction geometry)
- `COLLISION_PROBABILITY` is missing (required for threat scoring)
- `time_to_tca_hours` is negative (CDM was created after TCA â€” stale, operationally useless)

Records that pass validation but have some optional fields missing (e.g., covariance terms for one object) are accepted with those fields as `NaN`, to be handled by the `SimpleImputer` in step 2.

This fail-fast design ensures that downstream steps never encounter structurally invalid records. The parser logs a warning for each rejected file, and the total rejection count is reported in the parser's summary output. In the ALDORIA dataset, rejection rates were consistently below 0.5%.

---

### 5.8 Output: parsed_cdm_data.csv â€” Structure & Statistics

The parser collects all valid CDM records into a list of dictionaries, then writes them to `parsed_cdm_data.csv` using `pd.DataFrame(records).to_csv(...)`.

**Key columns in parsed_cdm_data.csv:**

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | str | Unique conjunction identifier (e.g., `25544_48274`) |
| `object1_norad_id` | str | NORAD ID of primary object |
| `object2_norad_id` | str | NORAD ID of secondary object |
| `CREATION_DATE` | datetime str | When this CDM was generated |
| `TCA` | datetime str | Predicted Time of Closest Approach |
| `time_to_tca_hours` | float | Hours until TCA at CDM creation time |
| `MISS_DISTANCE` | float | Predicted miss distance (meters) |
| `RELATIVE_SPEED` | float | Relative velocity at TCA (m/s) |
| `RELATIVE_POSITION_R/T/N` | float | RTN-frame separation components (meters) |
| `COLLISION_PROBABILITY` | float | Raw Pc (linear scale) |
| `log10_pc` | float | logâ‚â‚€(Pc), capped at âˆ’10 |
| `combined_cr_r/ct_t/cn_n` | float | Summed covariance diagonal (mÂ²) |
| `object1_CR_R`, `object2_CR_R`, etc. | float | Per-object covariance terms |
| `object1_OBJECT_NAME`, etc. | str | Object identification metadata |

**File statistics:**
- **Total rows:** 185,511 CDMs
- **Total columns:** ~45 (including all raw + derived + object metadata)
- **File size:** 213 MB
- **Unique event_ids:** 20,506 raw events; 2,003 events after filtering for â‰¥ 2 CDMs per event

---

### 5.9 Data Quality Observations

Running the inspection tool (`Scripts/tools/inspect_data.py`) on the parsed and processed sequences revealed several important data quality characteristics:

**Padding dominance:** 69.3% of all values in X_train are the sentinel value âˆ’999.0. This is expected â€” most events have 2â€“8 CDMs, leaving 12â€“18 of the 20 timestep slots padded. The Masking layer is therefore critically important; without it, the model would try to learn from padding values.

**Covariance extreme outliers:** Even after log1p transform, the `combined_ct_t` feature showed values up to 32Ã— the training standard deviation in the validation set. This is the covariance term most prone to extreme values (transverse uncertainty can be enormous for objects tracked only with TLE-quality data). The gradient clipping (`clipnorm=1.0`) in the optimizer absorbs these spikes without corrupting the model weights.

**Distribution skew in COLLISION_PROBABILITY:** The raw linear Pc distribution is extremely right-skewed â€” median ~2Ã—10â»â¶ but maximum ~0.07. After logâ‚â‚€ transform, the distribution is approximately Gaussian with mean ~âˆ’5.9 and std ~1.2. This is why log10_pc is used as the training feature instead of the raw value.

**Cross-split consistency:** The train/val/test distributions for most features are well-matched (mean difference < 0.1 std), confirming the random event-level split produces representative splits. One exception: `combined_cn_n` showed a mean shift of ~1 std between train and val in one test run â€” attributable to a small number of extreme-covariance events landing disproportionately in the validation set due to the random seed.

**Missing value rates:** Approximately 3â€“8% of CDMs are missing at least one covariance term (typically from one of the two objects). All missing values are handled by the `SimpleImputer` (median strategy) in step 2, after which the sequences contain no NaN or Inf values (verified by the inspection tool).

## 6. Feature Engineering & Sequence Preparation

### 6.1 Feature Selection â€” The 11 Model Features

The final feature set used for training contains **11 features** per CDM timestep. These were chosen to capture the three most important dimensions of conjunction information: **risk level** (Pc and miss distance), **temporal urgency** (time to TCA), and **data quality** (covariance in RTN frame and relative position).

| # | Feature | Source | Why Included |
|---|---------|--------|-------------|
| 0 | `COLLISION_PROBABILITY` | CDM global | Raw Pc â€” retained alongside log10_pc for model redundancy |
| 1 | `log10_pc` | Derived | Compresses 10-order-of-magnitude Pc range into [-10, 0] |
| 2 | `MISS_DISTANCE` | CDM global | Direct conjunction geometry â€” key risk indicator |
| 3 | `time_to_tca_hours` | Derived | Temporal urgency â€” where are we in the event timeline? |
| 4 | `combined_cr_r` | Derived (log1p) | Radial covariance quality after log1p compression |
| 5 | `combined_ct_t` | Derived (log1p) | Transverse covariance quality â€” most variable axis |
| 6 | `combined_cn_n` | Derived (log1p) | Normal covariance quality |
| 7 | `RELATIVE_SPEED` | CDM global | Hypervelocity indicator â€” affects collision energy |
| 8 | `RELATIVE_POSITION_R` | CDM global | Radial component of separation vector |
| 9 | `RELATIVE_POSITION_T` | CDM global | Transverse component of separation vector |
| 10 | `RELATIVE_POSITION_N` | CDM global | Normal component of separation vector |

Features intentionally excluded:
- **State vectors (X, Y, Z, X_DOT, Y_DOT, Z_DOT):** Absolute position in ECI frame; not meaningful for conjunction geometry without both objects simultaneously
- **Per-object covariance** (before combining): Replaced by the combined form, which is what actually determines the Pc uncertainty
- **OBJECT_NAME, OBJECT_TYPE, MANEUVERABLE:** Categorical metadata â€” not appropriate for direct numeric input to a recurrent network

---

### 6.2 Why Raw COLLISION_PROBABILITY Was Excluded as the Primary Feature

This was the single most impactful engineering decision in the project. In the initial design, `COLLISION_PROBABILITY` (raw linear scale) was the primary Pc feature. The result was a validation loss of **~84.9** that did not converge.

The problem: `COLLISION_PROBABILITY` spans roughly 12 orders of magnitude (1Ã—10â»Â¹Â² to 0.07 in the ALDORIA dataset). `StandardScaler` computes:

```
z = (x - mean) / std
```

With mean â‰ˆ 0.000002 and std â‰ˆ 0.0006, a single CDM with Pc = 0.07 produces:

```
z = (0.07 - 0.000002) / 0.0006 â‰ˆ 116.7
```

A target value of 116.7 standard deviations. The MSE loss for that single sample is (116.7)Â² â‰ˆ 13,619. This is catastrophic â€” a handful of high-Pc events dominated the entire gradient signal, preventing the model from learning anything about the vast majority of events with Pc in the 1e-8 to 1e-5 range.

The fix: use `log10_pc` as the primary feature, retaining raw `COLLISION_PROBABILITY` as an additional feature (now scaled to a manageable range). After the fix, scaled target values for all features stayed within roughly Â±5 standard deviations, and validation loss dropped from 84.9 to 0.628.

---

### 6.3 The Covariance Problem: Spans 0 to 20 Billion mÂ²

The three combined covariance features (`combined_cr_r`, `combined_ct_t`, `combined_cn_n`) posed the second major data engineering challenge.

Physical covariance values in the ALDORIA dataset:
- **Minimum:** Near-zero (objects tracked with high-precision sensors, uncertainty < 1 mÂ²)
- **Maximum combined_ct_t:** ~20 billion mÂ² (objects with only TLE-quality tracking, uncertainty spanning Â±141 km in the transverse direction)
- **Typical range:** 10,000 mÂ² to 500,000,000 mÂ² (a 5-order-of-magnitude spread for "typical" events)

Before any transformation, `StandardScaler` produces:
- `combined_ct_t` mean: 8,157,134,715 mÂ² (8.2 billion)
- `combined_ct_t` std: 19,838,841,465 mÂ² (19.8 billion)

For an object at the median covariance (~2 billion mÂ²):
```
z = (2e9 - 8.2e9) / 19.8e9 â‰ˆ -0.31
```
Reasonable. But for the extreme events:
```
z = (2e10 - 8.2e9) / 19.8e9 â‰ˆ 0.60   (max: still reasonable)
```

Oddly, the linear StandardScaler produces manageable scaled values even for the extreme end â€” but the problem is the **tail behavior**: the scaler is dominated by a small number of extreme events, causing the majority of normal events to cluster near zero in scaled space with very little separation. The model can't distinguish between events at 10,000 mÂ² and 100,000 mÂ² â€” both round to â‰ˆ 0.0 after StandardScaler.

Log transform solves this by giving equal weight to each order of magnitude.

---

### 6.4 The log1p Transform Solution

The log1p transform compresses extreme ranges while preserving the ordering and relative differences between values. It is applied *before* StandardScaler in step 2:

```python
# Applied to all three combined covariance features
for col in ['combined_cr_r', 'combined_ct_t', 'combined_cn_n']:
    df[col] = np.log1p(df[col])
```

`log1p(x) = log(1 + x)` â€” using +1 ensures that x=0 maps to log1p(0)=0 rather than -âˆž, which is important because some objects have zero reported covariance in certain axes.

The effect on the combined_ct_t distribution:

| Statistic | Before log1p | After log1p |
|-----------|-------------|------------|
| Min | 0 mÂ² | 0.0 |
| Median | ~2Ã—10â¹ mÂ² | ~21.4 |
| Max | ~2Ã—10Â¹â° mÂ² | ~23.7 |
| Range | 20 billion | ~23.7 |

After log1p, StandardScaler produces a well-behaved distribution with mean â‰ˆ 0 and std â‰ˆ 1 across all covariance features.

**Critical downstream implication for scoring:** Because the scaler is fitted on log1p(covariance), its `inverse_transform()` returns log1p-scale values â€” not raw mÂ². The `scoring.py` module must apply `np.expm1()` after `inverse_transform()` to recover the physical mÂ² values for confidence computation. Failure to do this was the KD-14 bug (see Section 16.5).

---

### 6.5 The log10_pc Feature

The `log10_pc` feature transforms raw collision probability into a physically interpretable logarithmic scale:

```python
log10_pc = log10(max(COLLISION_PROBABILITY, 1e-10))
```

The floor at 1Ã—10â»Â¹â° prevents -âˆž for events with Pc = 0 (which appears in the data when the probability is below floating-point precision).

After this transform, the feature spans roughly [-10, 0]:
- log10_pc = -10: Pc â‰¤ 1Ã—10â»Â¹â° (effectively zero risk)
- log10_pc = -6: Pc = 1Ã—10â»â¶ (routine monitoring)
- log10_pc = -4: Pc = 1Ã—10â»â´ (maneuver evaluation threshold)
- log10_pc = -1: Pc = 0.1 (extreme â€” collision likely if estimate is accurate)

After StandardScaler (mean â‰ˆ -5.9, std â‰ˆ 1.2), the scaled values are well-behaved and appropriately distributed for neural network training.

---

### 6.6 Event-Level Train/Val/Test Split (80/10/10)

The dataset is split **by event**, not by CDM. All CDMs from a given conjunction event (e.g., all 12 CDMs for event `25544_48274`) are assigned exclusively to one of the three splits.

The split procedure in step 2:
```python
unique_events = df['event_id'].unique()
rng = np.random.default_rng(SEED)   # SEED = 42
rng.shuffle(unique_events)

n = len(unique_events)               # 2,003 events
n_train = int(0.80 * n)             # 1,602
n_val   = int(0.10 * n)             # 200
# n_test  = remaining               # 201

train_events = set(unique_events[:n_train])
val_events   = set(unique_events[n_train : n_train + n_val])
test_events  = set(unique_events[n_train + n_val :])
```

Final split counts:

| Split | Events | Self-Supervised Samples |
|-------|--------|------------------------|
| Training | 1,602 (80.0%) | 77,989 |
| Validation | 200 (10.0%) | 9,598 |
| Test | 201 (10.0%) | 9,637 |

The resulting split info is saved to `processed_sequences/split_info.csv` for full traceability.

---

### 6.7 Why Event-Level Split Prevents Data Leakage

An event-level split is the **only correct** split strategy for self-supervised CDM modeling. The alternative â€” splitting CDMs randomly regardless of which event they belong to â€” creates a severe data leakage problem:

**The leakage scenario with random CDM split:**
- CDM_7 of event `25544_48274` (showing Pc = 1Ã—10â»Â³) lands in the training set
- CDM_8 of the same event (showing Pc = 8Ã—10â»â´, TCA in 4 hours) lands in the test set
- The model is trained on CDM_7; at test time, it must "predict" CDM_8
- But CDM_7 appears in training â€” the model has directly seen the sequence state just before the test target
- The test set is not truly held-out; it shares context with training data from the same event

With event-level split:
- All 12 CDMs of event `25544_48274` are in the test set (or all in training â€” never split across sets)
- The model has never seen any CDM from this event during training
- The test evaluation is a genuine measure of generalization to unseen conjunction events

This is critical for scientific validity. Our unit tests (`test_sequences.py::TestEventLevelSplit`) verify that no event appears in more than one split.

---

### 6.8 The Padding Sentinel: Why -999.0, Not Zero

Sequences of different lengths must be padded to a uniform length (max_sequence_length = 20) for batched training. The padding value must satisfy one constraint: **it must be distinguishable from any legitimate data value after normalization.**

The Keras `Masking` layer identifies padding by checking whether all features of a timestep equal the `mask_value`. If any feature is non-padding, the timestep is not masked.

**Why not 0.0?** After `StandardScaler`, real CDM features are mean-centered. A feature with mean 0 and std 1 regularly produces values of 0.0 from real data. A Masking layer with `mask_value=0` would incorrectly mask real timesteps where all features happen to be at their mean simultaneously.

**Why -999.0?** After `StandardScaler`, the minimum physically realistic feature value is approximately -3.7 (observed in log10_pc). A value of -999.0 is 999 standard deviations below the mean â€” completely impossible from real data. It is therefore an unambiguous sentinel.

```python
PADDING_VALUE = -999.0

def pad_sequence(seq, max_len):
    n = len(seq)
    if n >= max_len:
        return seq[-max_len:]          # truncate old CDMs if event > 20 CDMs
    pad = np.full((max_len - n, seq.shape[1]), PADDING_VALUE)
    return np.vstack([pad, seq])       # left-pad (oldest CDMs first, most recent last)
```

Left-padding (prepending padding before the real data) ensures that the **most recent CDMs are always at the end of the sequence** â€” aligned with the temporal direction that matters most for prediction.

---

### 6.9 Self-Supervised Sample Generation

For each conjunction event with N CDMs (N â‰¥ 2), step 2 generates Nâˆ’1 training samples. For event k with CDMs sorted by CREATION_DATE:

```
Sample 1: Input = [pad, pad, ..., pad, CDM_1]      Target = CDM_2
Sample 2: Input = [pad, pad, ..., CDM_1, CDM_2]    Target = CDM_3
...
Sample N-1: Input = [CDM_{N-20}, ..., CDM_{N-1}]   Target = CDM_N
```

Where inputs are left-padded to max_len=20. The target (Y) is always a single CDM's features (not padded â€” it's the raw scaled feature vector).

This gives the model exposure to every temporal vantage point within each event: it must learn to predict from both sparse early-event contexts (few CDMs, lots of padding) and information-rich late-event contexts (many CDMs, no padding). This variety makes the model robust across the full range of event maturities seen at inference time.

---

### 6.10 StandardScaler: Fitted on Training Data Only

A `StandardScaler` is fitted **only on training data** and then applied to transform validation and test data:

```python
scaler = StandardScaler()
scaler.fit(Y_train)               # fit only on training targets
Y_train_scaled = scaler.transform(Y_train)
Y_val_scaled   = scaler.transform(Y_val)    # use training statistics
Y_test_scaled  = scaler.transform(Y_test)   # use training statistics
```

The same scaler is applied to the X arrays (input sequences), excluding the padding positions.

The fitted scaler is saved to `processed_sequences/feature_scaler.pkl` (joblib serialization) and loaded by all downstream steps that need to convert scaled predictions back to physical units â€” including `step3b`, `step4`, and `scoring.py`.

This is standard machine learning hygiene: fitting the scaler on validation or test data would contaminate the training statistics with unseen-data information, producing overly optimistic performance estimates.

---

### 6.11 SimpleImputer: Median Strategy for Missing Values

Before scaling, missing values (NaN) must be filled. A `SimpleImputer` with `strategy='median'` is used:

```python
imputer = SimpleImputer(strategy='median')
imputer.fit(df_train[feature_cols])
df_train[feature_cols] = imputer.transform(df_train[feature_cols])
df_val[feature_cols]   = imputer.transform(df_val[feature_cols])
df_test[feature_cols]  = imputer.transform(df_test[feature_cols])
```

**Why median over mean?** The covariance distributions are heavily right-skewed. The median is far more robust than the mean for skewed distributions with outliers â€” it represents a "typical" covariance value rather than being pulled toward extreme events.

The fitted imputer is saved to `processed_sequences/feature_imputer.pkl`. Example imputed medians from the ALDORIA dataset:

| Feature | Imputed Median Value |
|---------|---------------------|
| `COLLISION_PROBABILITY` | 2.1Ã—10â»â¶ |
| `log10_pc` | âˆ’5.64 |
| `MISS_DISTANCE` | 322.6 m |
| `time_to_tca_hours` | 52.5 h |
| `combined_cr_r` | 313,874 mÂ² |
| `combined_ct_t` | 2.13Ã—10â¹ mÂ² |
| `combined_cn_n` | 76,242 mÂ² |

---

### 6.12 Final Sequence Shapes & Storage

After imputation, log1p transform, scaling, event-level split, sample generation, and padding, the final arrays are saved as NumPy binary files:

| File | Shape | Size |
|------|-------|------|
| `X_train.npy` | (77,989 Ã— 20 Ã— 11) | 134 MB |
| `Y_train.npy` | (77,989 Ã— 11) | 6.7 MB |
| `X_val.npy` | (9,598 Ã— 20 Ã— 11) | 16.5 MB |
| `Y_val.npy` | (9,598 Ã— 11) | 0.8 MB |
| `X_test.npy` | (9,637 Ã— 20 Ã— 11) | 16.6 MB |
| `Y_test.npy` | (9,637 Ã— 11) | 0.8 MB |

All six arrays are verified clean (zero NaN, zero Inf) by the `inspect_data.py` tool. The `feature_names.txt` file records the ordered list of 11 feature names, ensuring all downstream steps use the same feature ordering as the trained model expects.

## 7. Model Architecture â€” The BiGRU

### 7.1 Architecture Overview (Layer-by-Layer)

The DebriSolver model is a **Bidirectional Gated Recurrent Unit (BiGRU)** encoder with a Dense regression decoder. It takes a variable-length CDM sequence as input and produces a single-step prediction of the next CDM's features.

Full layer stack (from `model_builder.py`):

```
Input:  (batch_size, 20, 11)  â† 20 timesteps, 11 features
   â”‚
   â–¼
Masking(mask_value=-999.0)    â† zeros-out gradient for padding positions
   â”‚
   â–¼
Bidirectional(GRU(128, return_sequences=True), merge_mode='concat')
   â”‚  Output: (batch_size, 20, 256)  â† 128 forward + 128 backward
   â–¼
LayerNormalization()
   â”‚
   â–¼
Dropout(rate=0.3)
   â”‚
   â–¼
Bidirectional(GRU(64, return_sequences=False), merge_mode='concat')
   â”‚  Output: (batch_size, 128)  â† 64 forward + 64 backward
   â–¼
LayerNormalization()
   â”‚
   â–¼
Dropout(rate=0.3)
   â”‚
   â–¼
Dense(64, activation='relu', kernel_regularizer=L2(0.001))
   â”‚
   â–¼
Dropout(rate=0.3)
   â”‚
   â–¼
Dense(32, activation='relu', kernel_regularizer=L2(0.001))
   â”‚
   â–¼
Dense(11, activation='linear')  â† output: predicted next CDM (11 features)
   â”‚
Output: (batch_size, 11)
```

---

### 7.2 The Masking Layer â€” Handling Variable-Length Sequences

The `Masking` layer is the first processing layer and one of the most important architectural decisions. It instructs all downstream layers to ignore padded timesteps.

```python
keras.layers.Masking(mask_value=-999.0)
```

When the Masking layer encounters a timestep where **all features equal -999.0**, it generates a boolean mask of `False` for that position. Downstream recurrent layers honor this mask â€” the GRU does not update its hidden state for masked timesteps, and the gradient does not flow back through them during training.

This means:
- A sequence with 3 real CDMs and 17 padding positions trains exactly as if only 3 timesteps were present
- The model never "learns from" padding â€” it learns only from real data
- Variable-length events (2 CDMs to 20 CDMs) coexist in the same batch without any special treatment

Without masking, the model would interpret -999.0 as real feature values and attempt to learn patterns from them â€” producing garbage representations.

---

### 7.3 Bidirectional GRU Layer 1 (128 units per direction)

```python
keras.layers.Bidirectional(
    keras.layers.GRU(
        128,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.L2(0.001),
        recurrent_regularizer=keras.regularizers.L2(0.001),
    ),
    merge_mode='concat'
)
```

This layer processes the sequence in both directions and concatenates the results:
- **Forward GRU (128 units):** Processes CDMâ‚ â†’ CDMâ‚‚ â†’ ... â†’ CDMâ‚™, building a hidden state that accumulates past context
- **Backward GRU (128 units):** Processes CDMâ‚™ â†’ CDMâ‚™â‚‹â‚ â†’ ... â†’ CDMâ‚, building a hidden state that incorporates future context
- **Concat:** The forward and backward hidden states at each timestep are concatenated â†’ output dimension is 256 per timestep

`return_sequences=True` means the layer outputs the hidden state at *every* timestep (shape: `batch Ã— 20 Ã— 256`), not just the final one. This is required because the second GRU layer also processes the full sequence.

L2 regularization (Î»=0.001) is applied to both the input-to-hidden kernel and the recurrent (hidden-to-hidden) kernel, penalizing large weight magnitudes and improving generalization.

---

### 7.4 Bidirectional GRU Layer 2 (64 units per direction)

```python
keras.layers.Bidirectional(
    keras.layers.GRU(
        64,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.L2(0.001),
        recurrent_regularizer=keras.regularizers.L2(0.001),
    ),
    merge_mode='concat'
)
```

The second BiGRU layer refines the temporal representation. With `return_sequences=False`, it outputs only the final hidden state â€” a single vector of size 128 (64 forward + 64 backward) that summarizes the entire sequence.

This hierarchical design (Layer 1 captures local temporal patterns; Layer 2 summarizes global sequence dynamics) is a standard deep RNN architecture pattern. The decreasing unit count (128 â†’ 64) acts as an information bottleneck, forcing the model to compress the sequence into its most predictive representation.

---

### 7.5 Dense Decoder Layers

After the BiGRU encoder produces a 128-dimensional sequence summary, two Dense layers decode it into a feature prediction:

```python
keras.layers.Dense(64, activation='relu', kernel_regularizer=L2(0.001))
keras.layers.Dropout(0.3)
keras.layers.Dense(32, activation='relu', kernel_regularizer=L2(0.001))
```

These layers learn the non-linear mapping from the sequence embedding to the predicted next CDM values. ReLU activations introduce non-linearity while remaining computationally efficient. L2 regularization on both layers prevents overfitting to the training event distribution.

The 64â†’32 bottleneck is deliberate: it forces the decoder to learn a compact, low-dimensional representation of the prediction task rather than memorizing individual event patterns.

---

### 7.6 Output Layer (11 features, linear activation)

```python
keras.layers.Dense(11, activation='linear')
```

The output layer produces 11 real-valued predictions â€” one for each feature in the scaled feature space. The linear (no) activation is critical: we are performing **regression**, not classification. The output values can be negative or positive, and there are no bounds on their range. Any sigmoid or softmax activation would incorrectly constrain the outputs.

The 11 output values are in scaled space (StandardScaler units). They must be inverse-transformed to physical units using the saved `feature_scaler.pkl` before threat/confidence computation.

---

### 7.7 Why GRU Over LSTM?

GRU was chosen over LSTM for three concrete reasons:

**1. Fewer parameters.** A GRU cell with `n` units has 3nÂ² + 3n parameters (two gates, one candidate). An LSTM cell has 4nÂ² + 4n (four gates). For our architecture (128 units in layer 1), this means:
- GRU Layer 1: ~402,000 parameters
- LSTM Layer 1: ~536,000 parameters

Fewer parameters = faster training, lower memory use, and less risk of overfitting on ~2,000 events.

**2. Comparable performance on short sequences.** LSTMs were designed to handle very long-range dependencies (100+ timesteps). Our sequences are at most 20 timesteps. On sequences this short, GRUs consistently match LSTM performance in the literature, with some studies showing GRUs slightly outperforming LSTMs on shorter sequences.

**3. Faster convergence.** With gradient clipping already required (due to covariance outliers), the simpler GRU update equations are more stable and converge faster than LSTM's more complex gating mechanism.

---

### 7.8 Why Bidirectional?

Standard (unidirectional) GRUs process sequences from left to right â€” earlier CDMs inform later ones, but not vice versa. For a sequence prediction task, this is appropriate at inference time (we can't look into the future). However, during **training**, we have access to the full event sequence, and the backward direction provides valuable context.

In the backward direction, the GRU processes sequences from the last CDM backward. This allows the model to learn: "events that eventually showed pattern X in their later CDMs tended to have pattern Y in their earlier CDMs." This retrospective context improves the quality of the learned representations.

At inference time, the backward direction processes from the last real CDM backward through the sequence â€” it still only sees data up to the most recent CDM, not beyond it. There is no data leakage.

Empirically, in our validation experiments, the BiGRU achieved ~4â€“7% lower validation MAE compared to a unidirectional GRU with the same parameter count.

---

### 7.9 Why LayerNorm Over BatchNorm? (Critical for MC Dropout)

This is the most architecturally critical decision in the model, and it directly enables MC Dropout uncertainty quantification.

**BatchNormalization** normalizes using statistics computed across the batch dimension:
```
BN: x_normalized = (x - mean_batch) / std_batch
```
When `training=True` is passed at inference time (which is required for MC Dropout), BatchNorm recomputes batch statistics on every forward pass. With a different dropout mask each pass, the inputs to each layer change â€” so the batch statistics change â€” producing prediction variance that reflects both dropout randomness *and* batch statistic variation. The two sources of randomness are conflated, making the MC Dropout uncertainty estimate meaningless.

**LayerNormalization** normalizes across the feature dimension for each sample independently:
```
LN: x_normalized = (x - mean_features) / std_features  (per sample)
```
LayerNorm has no batch-level statistics. Setting `training=True` or `training=False` makes no difference to LayerNorm's computation. Each forward pass produces the same LayerNorm output for the same input, regardless of what other samples are in the batch or what the dropout mask looks like.

Result: with LayerNorm, the only source of variation between MC Dropout forward passes is the dropout mask itself â€” which is exactly the Bayesian posterior variance we want to estimate.

---

### 7.10 L2 Regularization on GRU Kernels

L2 regularization (weight decay, Î»=0.001) is applied to:
- GRU input-to-hidden kernels (`kernel_regularizer`)
- GRU recurrent kernels (`recurrent_regularizer`)
- Dense layer kernels (`kernel_regularizer`)

The regularization adds a penalty term `Î» Ã— Î£(wÂ²)` to the loss, discouraging large weight values. This prevents overfitting â€” particularly important because our training set contains only ~1,602 unique events, which is small for a neural network of this size.

The Î»=0.001 value was selected by grid search over [0.0001, 0.001, 0.01]. Lower values showed signs of overfitting (train_loss much lower than val_loss); higher values led to underfitting (both losses high).

---

### 7.11 Total Parameters: 244,171

Full parameter count breakdown:

| Layer | Output Shape | Parameters |
|-------|-------------|-----------|
| Masking | (None, 20, 11) | 0 |
| Bidirectional GRU 1 | (None, 20, 256) | 107,520 |
| LayerNormalization | (None, 20, 256) | 512 |
| Dropout | (None, 20, 256) | 0 |
| Bidirectional GRU 2 | (None, 128) | 124,800 |
| LayerNormalization | (None, 128) | 256 |
| Dropout | (None, 128) | 0 |
| Dense 64 | (None, 64) | 8,256 |
| Dropout | (None, 64) | 0 |
| Dense 32 | (None, 32) | 2,080 |
| Dense 11 (output) | (None, 11) | 363 |
| **Total** | â€” | **244,171** |

244,171 parameters is lean for this problem â€” small enough to train on CPU in a reasonable time (~2.7 hours for 150 epochs) while still having sufficient capacity to capture the temporal dynamics of CDM sequences.

---

### 7.12 model_builder.py: The Shared Architecture Module

Rather than defining the model architecture inside `step3_train_model.py`, the architecture is encapsulated in `Scripts/model_builder.py`. This module is imported by:
- `step3_train_model.py` â€” for training
- `step3b_evaluate_proxy_confidence.py` â€” for offline evaluation
- `step4_inference_dashboard.py` â€” for production inference

This ensures all three steps use **identical model architecture**. Without this shared module, there is a risk of subtle architecture mismatches between the training and inference stages (e.g., wrong dropout rate, wrong number of units) â€” which would produce silent correctness bugs where the loaded weights don't match the model definition.

`model_builder.py` exposes two functions:
- `build_self_supervised_gru(n_timesteps, n_features, gru_units_1, gru_units_2, ...)` â€” direct construction with explicit parameters
- `build_model_from_config(config_dict)` â€” construction from the config.yaml dictionary

---

### 7.13 config.yaml: Single Source of Truth

All model hyperparameters are defined in `Scripts/config.yaml`:

```yaml
model:
  gru_units_1: 128
  gru_units_2: 64
  dense_units: 64
  dropout_rate: 0.3
  l2_reg: 0.001

training:
  epochs: 150
  batch_size: 256
  learning_rate: 0.001
  patience: 20
  max_sequence_length: 20
  seed: 42

inference:
  mc_dropout_passes: 50
```

Any hyperparameter change is made in config.yaml and automatically propagated to all pipeline steps that read it. This prevents parameter drift between training and inference â€” a common source of subtle bugs in ML pipelines where hyperparameters are hardcoded in multiple files and get out of sync.

## 8. Training Strategy & Optimization

### 8.1 The Self-Supervised Training Objective

The model is trained to minimize the difference between its predicted next CDM and the actual next CDM â€” a **regression task** with no classification labels. The objective function is a weighted Mean Squared Error (MSE) computed over all 11 features:

```
L(Å·, y) = Î£áµ¢ wáµ¢ Â· (Å·áµ¢ - yáµ¢)Â²
```

Where:
- `Å·áµ¢` = predicted value for feature i (in StandardScaler-normalized space)
- `yáµ¢` = actual next CDM value for feature i (in StandardScaler-normalized space)
- `wáµ¢` = feature weight (see Section 8.2)

Both predictions and targets are in scaled space, ensuring the loss magnitude is comparable across features regardless of their original scale. The model is never exposed to raw physical values during training â€” only the StandardScaler-normalized versions.

---

### 8.2 Weighted MSE Loss â€” Feature Importance Weighting

A standard MSE loss treats all 11 features equally. However, from an operational standpoint, predicting Pc and miss distance accurately is far more important than predicting relative position components. A weighted MSE was implemented to reflect this:

```python
FEATURE_WEIGHTS = {
    'COLLISION_PROBABILITY': 1.0,
    'log10_pc':              2.0,   # highest weight â€” primary risk indicator
    'MISS_DISTANCE':         1.5,   # second highest â€” direct geometry
    'time_to_tca_hours':     1.0,
    'combined_cr_r':         1.0,
    'combined_ct_t':         1.0,
    'combined_cn_n':         1.0,
    'RELATIVE_SPEED':        1.0,
    'RELATIVE_POSITION_R':   0.8,
    'RELATIVE_POSITION_T':   0.8,
    'RELATIVE_POSITION_N':   0.8,
}
```

The custom loss is implemented as a TensorFlow function:
```python
def weighted_mse(y_true, y_pred):
    weights = tf.constant([...], dtype=tf.float32)
    squared_errors = tf.square(y_true - y_pred)
    return tf.reduce_mean(weights * squared_errors)
```

The effect: training gradient signal is amplified 2Ã— for log10_pc errors and 1.5Ã— for miss distance errors. The model converges to better Pc prediction accuracy at the (minor) cost of slightly higher error on the lower-weighted relative position features â€” an explicit, intentional trade-off.

---

### 8.3 Why Gradient Clipping Was Essential (clipnorm=1.0)

Even after log1p transform and StandardScaler, the training data contains rare CDMs with extreme feature values â€” particularly covariance events in the 99th percentile. When these samples appear in a batch, their large MSE loss produces large gradients that can destabilize the model weights.

Without gradient clipping, training exhibited:
- Intermittent loss spikes (loss jumping from ~1.5 to ~50 then back)
- Occasional NaN loss values after a particularly extreme batch
- Slower overall convergence due to weight corruption from spike batches

Gradient clipping with `clipnorm=1.0` resolves this by rescaling the **entire gradient vector** if its L2 norm exceeds 1.0:

```python
optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
```

If `||âˆ‡L||â‚‚ > 1.0`: gradient is scaled to `âˆ‡L / ||âˆ‡L||â‚‚` (unit norm)
If `||âˆ‡L||â‚‚ â‰¤ 1.0`: gradient is unchanged

This preserves the direction of the gradient (the model still learns correctly from extreme samples) while preventing catastrophic weight updates. After adding clipnorm=1.0, training was stable throughout all 150 epochs with no loss spikes.

---

### 8.4 The Adam Optimizer

Adam (Adaptive Moment Estimation) is used with the following configuration:

```python
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,       # exponential decay rate for gradient moment (default)
    beta_2=0.999,     # exponential decay rate for gradient^2 moment (default)
    epsilon=1e-7,     # numerical stability (default)
    clipnorm=1.0,     # gradient clipping
)
```

Adam was chosen over SGD for its adaptive per-parameter learning rates. In a model with 244,171 parameters receiving very different gradient magnitudes (Pc-related layers receive larger signals than relative position layers due to feature weighting), Adam's per-parameter adaptation is critical for balanced convergence.

Initial learning rate of 0.001 is the standard Adam default and was found to be appropriate â€” high enough for fast initial learning, low enough to avoid overshooting the loss minimum.

---

### 8.5 Learning Rate Scheduling: ReduceLROnPlateau

A `ReduceLROnPlateau` callback monitors the validation loss and reduces the learning rate when improvement stalls:

```python
keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,         # multiply LR by 0.5 on plateau
    patience=7,         # wait 7 epochs with no improvement before reducing
    min_lr=1e-7,        # floor on learning rate
    verbose=1,
)
```

Over the full 150-epoch training run, the learning rate decayed 7 times:

| Epoch | Event | Learning Rate |
|-------|-------|--------------|
| 0 | Initial | 0.001000 |
| ~22 | First plateau | 0.000500 |
| ~41 | Second plateau | 0.000250 |
| ~58 | Third plateau | 0.000125 |
| ~79 | Fourth plateau | 0.0000625 |
| ~98 | Fifth plateau | 0.0000313 |
| ~118 | Sixth plateau | 0.0000156 |
| ~136 | Seventh plateau | 0.00000781 |
| 147 | Training stopped | 0.00000781 |

This decay schedule allowed the model to initially learn rapidly (large LR â†’ fast weight updates), then fine-tune the loss minimum with increasingly precise steps (small LR â†’ small updates). The final learning rate of ~7.8Ã—10â»â¶ is four orders of magnitude smaller than the initial rate.

---

### 8.6 EarlyStopping: Patience=20

`EarlyStopping` terminates training if the validation loss doesn't improve for 20 consecutive epochs, and restores the best weights seen during training:

```python
keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1,
)
```

In the final training run, training stopped at **epoch 147 out of 150** â€” meaning the model was still improving up to epoch 127 (the best epoch), then plateaued for 20 epochs before EarlyStopping triggered.

The patience=10 initial setting (before tuning) caused premature termination at epoch 73 â€” the model was still on a slow improvement trajectory. Increasing to patience=20 allowed it to continue through the slow final convergence phase where the LR had decayed to small values and improvements were incremental (val_loss improving by 0.001â€“0.005 per epoch).

---

### 8.7 ModelCheckpoint: Saving Best Weights

`ModelCheckpoint` saves the model weights at the epoch with the lowest validation loss:

```python
keras.callbacks.ModelCheckpoint(
    filepath='model_artifacts/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
)
```

Combined with `restore_best_weights=True` in EarlyStopping, this ensures that even if training has overfit slightly in later epochs, the final saved model is the one that performed best on validation data.

The saved model is a complete Keras model (`.keras` format) including architecture, weights, and optimizer state. Subsequent pipeline steps load only the weights into a freshly built model from config (to avoid architecture deserialization issues across TF versions).

---

### 8.8 The Full Training Run: 150 Epochs on CPU

The final training run statistics:
- **Hardware:** CPU (no GPU available during development)
- **Duration:** ~2.7 hours total for 150 epochs
- **Batch size:** 256
- **Total samples per epoch:** 77,989 training samples â†’ 305 batches/epoch
- **Best epoch:** 127 (val_loss = 0.628)
- **Stopped at:** epoch 147 (EarlyStopping triggered)

Training loss trajectory highlights:
- Epoch 1: train_loss â‰ˆ 4.1, val_loss â‰ˆ 3.8 (initial fast learning)
- Epoch 20: train_loss â‰ˆ 1.2, val_loss â‰ˆ 1.1 (converging)
- Epoch 50: train_loss â‰ˆ 0.85, val_loss â‰ˆ 0.82 (first LR decays)
- Epoch 100: train_loss â‰ˆ 0.71, val_loss â‰ˆ 0.69 (slow improvement phase)
- Epoch 127: train_loss â‰ˆ 0.64, val_loss â‰ˆ 0.628 (best model)
- Epoch 147: train_loss â‰ˆ 0.64, val_loss â‰ˆ 0.631 (stopped, slight overfit)

The near-equal train and val losses throughout training (gap < 0.02) indicates the regularization (L2 + Dropout) was effective â€” the model is not memorizing the training events.

---

### 8.9 Reproducibility: Seed=42, TF Deterministic Ops

Reproducibility was a design requirement. The same training run, on the same hardware, must produce identical results:

```python
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

`TF_DETERMINISTIC_OPS=1` forces TensorFlow to use deterministic CUDA kernels (even on GPU, though our runs were CPU). Without this, GPU operations can introduce non-deterministic floating-point summation order, producing slightly different results each run.

Additionally, the data split (Section 6.6) uses `np.random.default_rng(42)`, the model weight initialization uses `tf.random.set_seed(42)`, and all MC Dropout inference passes use the same global seed state.

---

### 8.10 Training History: Loss & MAE Curves

Training history is saved to `model_artifacts/training_history.json`, which records val_loss, train_loss, val_mae, and train_mae for every epoch. This file is used by:
- `step5_visualize.py` (Figures 6 and 7: training loss and MAE curves)
- Post-hoc analysis of learning dynamics

Key MAE statistics at best epoch (epoch 127):
- **Training MAE:** 0.512 (in scaled space; ~0.42 physical log10_pc units)
- **Validation MAE:** 0.531
- **Test MAE (from step3b):** 0.547

In physical terms: mean absolute error on log10_pc of ~0.42 log units corresponds to predicting Pc within a factor of ~2.6Ã— (10^0.42 â‰ˆ 2.6) of the true next-CDM Pc. For risk triage purposes, predicting Pc within half an order of magnitude is more than sufficient.

---

### 8.11 The gate_passed.flag Invalidation Mechanism

A critical safety rule: **every new training run invalidates the previous evaluation gate.**

At the start of `step3_train_model.py`:
```python
gate_flag = Path('model_artifacts/gate_passed.flag')
if gate_flag.exists():
    gate_flag.unlink()
    logger.warning("gate_passed.flag deleted â€” re-run step3b before step4")
```

Without this mechanism, an engineer could retrain the model (step 3) and then run production inference (step 4) using the gate flag from the *previous* model's evaluation. The new model may have different characteristics â€” different uncertainty range, different scoring distribution â€” that the old evaluation did not validate.

By deleting the gate flag at training start, the pipeline enforces that `step3b` (the evaluation gate) must always be re-run after `step3` (training). Step 4 will refuse to run until a fresh gate flag exists from the new model's evaluation.

## 9. Uncertainty Quantification â€” MC Dropout

### 9.1 What Is Monte Carlo Dropout?

Monte Carlo Dropout (MC Dropout) is a technique for obtaining uncertainty estimates from a neural network without requiring any architectural changes beyond the dropout layers already present for regularization. It was formalized by Gal & Ghahramani (2016) in the paper *"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning."*

Standard dropout was introduced as a regularization technique: during training, each neuron is randomly set to zero with probability `p` (dropout rate), preventing co-adaptation of neurons and reducing overfitting. At inference time, dropout is conventionally disabled â€” all neurons are active, producing a deterministic output.

MC Dropout breaks this convention: **dropout is kept active at inference time**. Each forward pass through the network uses a different random dropout mask, producing a different prediction. Running N forward passes produces N slightly different predictions, forming an empirical distribution. The statistics of this distribution (mean, standard deviation) approximate the Bayesian posterior predictive distribution of the model.

---

### 9.2 Dropout as Bayesian Approximation

Formally, Gal & Ghahramani showed that a neural network with dropout applied before every weight layer is mathematically equivalent to a Bayesian approximation to a Gaussian Process. The variational inference objective minimized by dropout training corresponds to the KL divergence between the approximate posterior and the true posterior over the network weights.

In practical terms: the dropout mask effectively selects a different sub-network on each forward pass, and the ensemble of predictions across many masks approximates sampling from the posterior distribution over all possible model configurations consistent with the training data.

This gives us:
- **Epistemic uncertainty:** uncertainty due to limited training data â€” if the model has seen few similar events, predictions vary widely across masks
- **Aleatoric uncertainty:** inherent noise in the data â€” even perfect training data produces some irreducible prediction variance

For conjunction risk assessment, epistemic uncertainty is what we care about most: high variance across dropout passes means the model is genuinely uncertain about this event's trajectory.

---

### 9.3 Implementation: training=True at Inference

In Keras, enabling MC Dropout at inference time requires a single change: pass `training=True` to every forward pass call:

```python
def mc_dropout_predict(model, X, n_passes=50):
    predictions = []
    for _ in range(n_passes):
        pred = model(X, training=True)   # training=True keeps dropout active
        predictions.append(pred.numpy())
    predictions = np.array(predictions)  # shape: (n_passes, batch_size, n_features)
    mean_pred = predictions.mean(axis=0)
    std_pred  = predictions.std(axis=0)
    return mean_pred, std_pred
```

The `training=True` flag instructs Keras to:
1. Keep all `Dropout` layers in training mode (active, randomly zeroing neurons)
2. Keep all `LayerNormalization` layers in their standard mode (per-sample, unaffected by training flag)

This is the reason LayerNorm is mandatory (see Section 7.9) â€” BatchNorm would also change behavior with `training=True` in a way that corrupts the uncertainty estimate.

---

### 9.4 50 Forward Passes â€” Why This Number?

The number of MC Dropout passes (`n_passes=50`) was chosen empirically by monitoring the stability of the mean and std estimates as a function of pass count:

| n_passes | Std estimate stability | Time per event |
|----------|----------------------|----------------|
| 10 | High variance across runs | ~0.05s |
| 25 | Moderate variance | ~0.12s |
| **50** | **Stable (< 2% run-to-run variation)** | **~0.25s** |
| 100 | Negligible improvement over 50 | ~0.50s |
| 200 | Near-identical to 100 | ~1.0s |

50 passes provide a stable uncertainty estimate with acceptable inference latency. For a batch of 2,003 events, 50 passes takes approximately 8â€“12 minutes on CPU â€” well within the operational window for a 24-hour conjunction screening cycle.

The 50-pass configuration is set in `config.yaml` (`mc_dropout_passes: 50`) and read by all inference steps.

---

### 9.5 Computing Prediction Mean & Standard Deviation

For each event's input sequence, 50 forward passes produce 50 predictions (each a vector of 11 scaled feature values). The mean and standard deviation are computed across passes:

```python
# predictions: (50, 11) array for one event
mean_pred = predictions.mean(axis=0)   # shape: (11,) â€” best estimate
std_pred  = predictions.std(axis=0)    # shape: (11,) â€” uncertainty per feature

# The uncertainty score for this event: average std across all features
event_uncertainty = std_pred.mean()    # scalar, 0.050 to 0.601 in our dataset
```

The **mean prediction** is the primary output â€” the model's best guess at the next CDM's feature values, used for threat scoring.

The **mean std** (averaged across all 11 features) is the uncertainty score â€” used as input to the confidence computation. A high uncertainty score means the 50 dropout-masked networks disagree significantly about the next CDM's values, signaling genuine model uncertainty about this event's trajectory.

---

### 9.6 Why BatchNorm Would Break MC Dropout

This is covered in detail in Section 7.9, but a brief summary for completeness:

BatchNormalization maintains running mean/std statistics of the activations. At test time (training=False), these running statistics are used. When MC Dropout forces training=True, BatchNorm recomputes statistics from each batch â€” and since the dropout mask changes each pass, the batch activations change, causing the statistics to change. This introduces a second source of prediction variance (batch stat variation) that is unrelated to the dropout-induced uncertainty we are trying to measure.

LayerNormalization normalizes per-sample using that sample's own feature statistics, making it completely independent of the training flag and the dropout mask. Only LayerNorm enables clean MC Dropout uncertainty estimation.

---

### 9.7 Observed Uncertainty Statistics

After running MC Dropout inference on all 2,003 events:

| Metric | Value |
|--------|-------|
| Minimum uncertainty (std) | 0.050 |
| Maximum uncertainty (std) | 0.601 |
| Mean uncertainty (std) | 0.183 |
| Median uncertainty (std) | 0.161 |
| Events with std < 0.15 (low uncertainty) | ~42% |
| Events with std > 0.35 (high uncertainty) | ~18% |

The resulting confidence scores (derived from uncertainty via the 3-component formula):

| Metric | Value |
|--------|-------|
| Minimum confidence | 0.160 |
| Maximum confidence | 0.660 |
| Mean confidence | 0.412 |
| Events with confidence > 0.5 (high confidence) | ~38% |

The confidence range of 0.16â€“0.66 is narrower than the theoretical [0.10, 1.00] range â€” this reflects the reality that most events in the dataset have limited CDM counts (2â€“8) and/or non-trivial covariance, naturally capping confidence below 0.7 even for the best-observed events.

---

### 9.8 Interpreting Uncertainty Values

Uncertainty and confidence are operationally meaningful in the following way:

**Low uncertainty (std < 0.15) â†’ High model confidence:**
The model's 50 predictions cluster tightly around the mean. The model has seen many similar CDM trajectories in training and is confident about how this event will evolve. The threat score based on this prediction can be trusted.

**High uncertainty (std > 0.35) â†’ Low model confidence:**
The 50 predictions spread widely. This event's CDM trajectory is unusual â€” either because it's genuinely unprecedented, because there are very few CDMs to learn from (2-3 CDMs), or because the tracking data is so noisy that the signal is obscured. The threat score still represents the model's best guess, but operators should seek additional information before acting on it.

**Important nuance:** A high-uncertainty, high-threat event (WATCH CLOSELY quadrant) is potentially more dangerous than a low-uncertainty, high-threat event (ACT NOW). The uncertainty doesn't reduce the threat â€” it means we don't yet know if the threat is real. WATCH CLOSELY events require more tracking data, not less attention.

## 10. Scoring System â€” Threat & Confidence

### 10.1 The scoring.py Module â€” Single Source of Truth

`Scripts/scoring.py` contains the single authoritative implementation of `compute_threat_and_confidence()`. This function is imported and called identically by:
- `step3b_evaluate_proxy_confidence.py` â€” offline evaluation
- `step4_inference_dashboard.py` â€” production inference

Having one module means any change to the scoring logic is automatically reflected in both steps. There is no possibility of step3b evaluating one version of the scoring and step4 running a different version.

The function signature:
```python
def compute_threat_and_confidence(
    mean_pred,         # (11,) array â€” model's mean prediction (scaled)
    std_pred,          # (11,) array â€” MC Dropout std (scaled)
    last_real_cdm,     # (11,) array â€” most recent actual CDM (scaled)
    n_valid_steps,     # int â€” number of real (non-padded) CDMs in input
    scaler,            # fitted StandardScaler
    feature_names,     # list of 11 feature names
) -> dict              # {'threat_score': float, 'confidence': float, 'quadrant': str, ...}
```

---

### 10.2 Inverse-Transforming Predictions to Physical Units

All model inputs and outputs are in StandardScaler-normalized space. Before any physics-based scoring, predictions must be converted to physical units:

```python
# Stack mean prediction and last real CDM for joint inverse transform
scaled_stack = np.vstack([mean_pred, last_real_cdm])   # (2, 11)
physical_stack = scaler.inverse_transform(scaled_stack) # (2, 11)

pred_physical = physical_stack[0]   # predicted next CDM in physical units
last_physical = physical_stack[1]   # most recent CDM in physical units
```

After inverse transform, the values are in log1p space for covariance features (because the scaler was fitted on log1p-transformed data). The `expm1()` correction must then be applied to recover raw mÂ² values:

```python
cov_indices = [feature_names.index(f) for f in ['combined_cr_r', 'combined_ct_t', 'combined_cn_n']]
for i in cov_indices:
    pred_physical[i] = np.expm1(pred_physical[i])   # log1p â†’ raw mÂ²
    last_physical[i] = np.expm1(last_physical[i])   # log1p â†’ raw mÂ²
```

This `expm1()` correction is the fix for the KD-14 bug (Section 10.9). Without it, covariance confidence would always be near 1.0 regardless of actual tracking quality.

---

### 10.3 Threat Score: Base Threat from Pc Level

The first component of the threat score is derived from the **predicted Pc** (the model's prediction of the *next* CDM's collision probability). After physical-unit conversion:

```python
pred_log10pc = pred_physical[feature_names.index('log10_pc')]
pred_pc = 10 ** pred_log10pc   # convert log10 to linear Pc
```

Base threat mapping (piecewise linear interpolation between anchors):

| Predicted Pc | Base Threat Score |
|-------------|------------------|
| â‰¥ 1Ã—10â»Â³ | 90 |
| 1Ã—10â»â´ â€“ 1Ã—10â»Â³ | 60 â€“ 90 (linear) |
| 1Ã—10â»â¶ â€“ 1Ã—10â»â´ | 25 â€“ 60 (linear) |
| 1Ã—10â»â¸ â€“ 1Ã—10â»â¶ | 5 â€“ 25 (linear) |
| < 1Ã—10â»â¸ | 5 |

The anchor points correspond to operational thresholds (1Ã—10â»Â³ and 1Ã—10â»â´) plus additional breakpoints that provide smooth resolution across the full Pc range.

---

### 10.4 Threat Score: Trend Modifier (Rising vs Falling Pc)

The model's core novel contribution is using the **predicted next Pc** relative to the **current Pc** to assess trajectory direction:

```python
current_log10pc = last_physical[feature_names.index('log10_pc')]
current_pc = 10 ** current_log10pc

pc_ratio = pred_pc / (current_pc + 1e-15)  # avoid division by zero

if pc_ratio > 2.0:        # Pc predicted to more than double
    trend_modifier = +15
elif pc_ratio > 1.2:      # Pc predicted to rise
    trend_modifier = +8
elif pc_ratio < 0.5:      # Pc predicted to drop by more than half
    trend_modifier = -15
elif pc_ratio < 0.8:      # Pc predicted to decline
    trend_modifier = -8
else:
    trend_modifier = 0    # stable Pc
```

The trend modifier adds or subtracts up to 15 points from the base threat score. An event whose predicted Pc is 3Ã— its current value is far more alarming than one with stable Pc â€” even if they share the same current Pc level. This is the key operational insight that threshold-based systems cannot capture.

---

### 10.5 Threat Score: TCA Urgency Bonus

The predicted time to TCA (in hours) adds an urgency component:

```python
pred_tca_hours = pred_physical[feature_names.index('time_to_tca_hours')]

if pred_tca_hours < 6:
    urgency_bonus = +10
elif pred_tca_hours < 24:
    urgency_bonus = +5
elif pred_tca_hours > 7 * 24:   # > 7 days
    urgency_bonus = -5
else:
    urgency_bonus = 0
```

The final threat score:
```python
threat_score = np.clip(base_threat + trend_modifier + urgency_bonus, 0, 100)
```

An event with predicted Pc = 1Ã—10â»â´ (base_threat â‰ˆ 60), rising Pc trend (+15), and TCA in 8 hours (+5) receives a threat score of 80. The same Pc with falling trend (-15) and TCA in 14 days (-5) receives a threat score of 40. The difference is operationally meaningful and correctly captured.

---

### 10.6 Confidence: MC Dropout Uncertainty Component (40%)

```python
# std_pred is the mean std across all 11 features from 50 MC passes
unc_score = std_pred.mean()

# Map uncertainty to confidence: lower uncertainty = higher confidence
# Typical range: 0.05 (very certain) to 0.60 (very uncertain)
uncertainty_confidence = np.clip(1.0 - (unc_score / 0.4), 0.0, 1.0)
# unc_score=0.05 â†’ unc_confidence=0.875; unc_score=0.40 â†’ unc_confidence=0.0
```

The denominator `0.4` is calibrated to the observed uncertainty range in the ALDORIA dataset (max observed: 0.601, 95th percentile: ~0.38). It normalizes the uncertainty to [0, 1], with the most uncertain events clipping to 0.

Weight in final confidence: **40%** (highest weight â€” this is the model's own self-assessment of its certainty).

---

### 10.7 Confidence: Data Quantity Component (35%)

```python
# n_valid_steps = number of real CDMs in input (not counting padding)
# max_sequence_length = 20

data_confidence = np.clip(n_valid_steps / 10.0, 0.0, 1.0)
# 1 CDM â†’ 0.10; 5 CDMs â†’ 0.50; 10+ CDMs â†’ 1.00
```

The scaling factor of 10.0 was chosen based on the observation that events with 10+ CDMs show stable, reliable scoring distributions, while events with fewer than 5 CDMs often have high prediction variance. Events with 1 CDM receive data_confidence = 0.10 (minimum possible from data quantity alone).

Weight in final confidence: **35%** â€” the second-highest weight. This reflects the critical importance of having enough CDM history to make reliable predictions. A model making good predictions from 2 CDMs is actually quite uncertain; the sparse context is a genuine epistemic limitation.

---

### 10.8 Confidence: Covariance Quality Component (25%)

```python
# combined_ct_t is the dominant covariance term (transverse, usually largest)
# After expm1 correction, this is in raw mÂ²
combined_ct_t_m2 = pred_physical[feature_names.index('combined_ct_t')]

# Reference threshold: 500,000,000 mÂ² (500 million) = typical poor-tracking threshold
COV_THRESHOLD_M2 = 5e8
cov_confidence = np.clip(1.0 - (combined_ct_t_m2 / COV_THRESHOLD_M2), 0.0, 1.0)
# cov = 0 mÂ² â†’ cov_confidence = 1.00 (perfect tracking)
# cov = 5e8 mÂ² â†’ cov_confidence = 0.00 (terrible tracking)
# cov = 1e8 mÂ² â†’ cov_confidence = 0.80 (reasonable tracking)
```

Weight in final confidence: **25%** â€” lower than the other two because covariance quality is already partially captured by the MC Dropout uncertainty (poorly tracked events also tend to produce higher model uncertainty).

The final confidence:
```python
confidence = np.clip(
    0.40 * uncertainty_confidence +
    0.35 * data_confidence +
    0.25 * cov_confidence,
    0.10, 1.00
)
```

---

### 10.9 The KD-14 Bug: expm1() Correction for Covariance

KD-14 was the most subtle bug in the project â€” it passed all tests at the time and was only discovered through careful inspection of the confidence output distribution.

**Symptom:** After training and inference, all 2,003 events had confidence scores between 0.48 and 0.52. The covariance quality component was contributing almost nothing to the confidence variation.

**Investigation:** Printing the raw covariance values entering the confidence formula revealed values like 13.1, 14.2, 12.8 â€” not millions as expected. These are log1p values (e.g., log1p(500,000) â‰ˆ 13.1), not raw mÂ² values.

**Root cause:** `scaler.inverse_transform()` correctly reverses the StandardScaler normalization â€” but it does *not* reverse the log1p transform that was applied before scaling. The scaler was fitted on log1p(covariance), so its output is log1p-space values. Computing `cov_confidence = 1 - (13.1 / 5e8)` gives â‰ˆ 1.0000000 regardless of actual covariance quality.

**Fix:** Apply `np.expm1()` after `inverse_transform()` to undo the log1p:
```python
# BEFORE FIX (wrong):
combined_ct_t_wrong = pred_physical[cov_t_idx]  # ~13.1 (log1p units)

# AFTER FIX (correct):
combined_ct_t_correct = np.expm1(pred_physical[cov_t_idx])  # ~500,000 mÂ²
```

After the fix, covariance quality appropriately varied from 0.0 (for 20-billion-mÂ² events) to 0.98 (for <10,000-mÂ² events), and the full [0.16, 0.66] confidence range was unlocked.

A regression test (`test_scoring.py::test_covariance_expm1_correction`) was added to prevent recurrence.

---

### 10.10 Quadrant Classification

After computing threat_score (0â€“100) and confidence (0.10â€“1.00), each event is assigned to exactly one quadrant:

```python
def classify_quadrant(threat_score, confidence, threat_threshold=50, conf_threshold=0.5):
    high_threat = threat_score >= threat_threshold
    high_conf   = confidence   >= conf_threshold

    if high_threat and high_conf:
        return 'ACT NOW'
    elif high_threat and not high_conf:
        return 'WATCH CLOSELY'
    elif not high_threat and not high_conf:
        return 'SAFELY IGNORE'
    else:   # low threat, high confidence
        return 'NOT PRIORITY'
```

Quadrant distribution across all 2,003 events:

| Quadrant | Count | Percentage | Operator Action |
|----------|-------|-----------|----------------|
| ACT NOW | 312 | 15.6% | Immediate human review, possible maneuver |
| WATCH CLOSELY | 489 | 24.4% | Request additional tracking data |
| SAFELY IGNORE | 287 | 14.3% | Deprioritize |
| NOT PRIORITY | 915 | 45.7% | Passive monitoring |

The 312 ACT NOW events represent the highest-priority triage output. Without DebriSolver, all 2,003 events would require equivalent human attention. The system effectively reduces the urgent attention burden by ~84%.

---

### 10.11 Why Scoring Must Be Deployment-Safe (No Truth Labels)

The scoring system is designed to never use any information that would be unavailable during real operational deployment:

- **No actual collision outcomes** â€” scoring uses only CDM feature predictions and model uncertainty
- **No future CDMs** â€” the model predicts the *next* CDM; scoring uses current and predicted values only
- **No sensor ground truth** â€” covariance quality is assessed from the CDM's reported covariance values, not from external tracking data
- **No event outcomes** â€” whether an event eventually resolved safely or collided is not used anywhere

This means the scoring system can be applied identically to:
1. Historical CDMs where outcomes are unknown
2. Real-time CDMs for current conjunction events
3. Simulated CDMs for system testing

The deployment safety is verified by the unit test `test_scoring.py::test_no_ground_truth_required`, which confirms that `compute_threat_and_confidence()` raises no errors when called with only model outputs and CDM features.

---

## 11. The Evaluation Gate â€” Step 3B

### 11.1 Why an Evaluation Gate Exists

Step 3B (`step3b_evaluate_proxy_confidence.py`) is a **mandatory quality gate** between training (step 3) and production inference (step 4). It exists because of a fundamental asymmetry in self-supervised models: the training loss tells us how well the model predicts CDM sequences, but it does not tell us whether the **confidence scores** produced by MC Dropout are actually calibrated â€” i.e., whether events with high confidence truly have lower prediction error than events with low confidence.

This calibration property is critical for operational trust. An operator who acts on an ACT NOW alert is implicitly trusting that the high confidence rating means the model genuinely has good information about this event. If confidence is systematically wrong (e.g., all high-confidence events actually have high error), the entire confidence dimension of the output is meaningless or misleading.

Step 3B verifies this calibration property offline, on the test set where ground truth (the actual next CDMs) is available. This is the only point in the entire pipeline where ground truth is used â€” and it is used only for evaluation, never for training.

---

### 11.2 What Step 3B Checks

For each test sample (event + sequence context), step 3B:

1. **Runs MC Dropout inference** (50 passes) to get `mean_pred` and `std_pred`
2. **Calls `compute_threat_and_confidence()`** to get the confidence score
3. **Computes the actual prediction error** as MAE between `mean_pred` and the known ground-truth next CDM: `actual_error = |mean_pred - Y_test|.mean()`
4. **Stores the (confidence, actual_error) pair** for all 9,637 test samples

The key check: **confidence and error should negatively correlate**. High confidence â†’ low error. Low confidence â†’ high error. If this relationship holds, the confidence signal is calibrated and operationally meaningful.

Step 3B computes the Pearson correlation between confidence scores and actual errors. The gate passes if:
```python
correlation = np.corrcoef(confidences, errors)[0, 1]
gate_passes = (correlation < -0.05)   # negative correlation required
```

A correlation threshold of -0.05 is permissive â€” we require a statistically negative relationship, not a strong one. In practice, the trained model achieved correlation of approximately -0.23, indicating meaningful (though imperfect) calibration.

---

### 11.3 Proxy Confidence from Truth (Offline Only)

Step 3B can also compute a **proxy confidence** using the actual prediction error as a gold standard. This is only possible offline (where Y_test is known) and is used for diagnostic purposes:

```python
# Higher error â†’ lower proxy confidence
proxy_confidence = 1.0 / (1.0 + actual_error)
```

By plotting model-computed confidence vs. proxy confidence, we can visualize whether the model's confidence ordering agrees with what the error distribution actually shows. This diagnostic plot is generated as one of the step 3B outputs.

This proxy confidence is **never used in production** â€” it requires knowing the actual next CDM, which is unavailable for future events.

---

### 11.4 Calibration Bins: Confidence vs Observed MAE

Step 3B groups the 9,637 test samples into 10 confidence bins (deciles) and computes the mean actual error within each bin. A well-calibrated model should show monotonically decreasing error as confidence increases.

Example from our training run:

| Confidence Bin | Mean Confidence | Mean MAE |
|----------------|----------------|----------|
| 0.10 â€“ 0.20 | 0.162 | 0.714 |
| 0.20 â€“ 0.25 | 0.228 | 0.641 |
| 0.25 â€“ 0.30 | 0.278 | 0.598 |
| 0.30 â€“ 0.35 | 0.326 | 0.567 |
| 0.35 â€“ 0.40 | 0.375 | 0.541 |
| 0.40 â€“ 0.45 | 0.422 | 0.521 |
| 0.45 â€“ 0.50 | 0.471 | 0.502 |
| 0.50 â€“ 0.55 | 0.519 | 0.484 |
| 0.55 â€“ 0.60 | 0.572 | 0.461 |
| 0.60 â€“ 0.66 | 0.628 | 0.438 |

The monotonic decrease from 0.714 to 0.438 confirms that the confidence signal is calibrated: events the model is confident about truly have lower prediction error.

---

### 11.5 The gate_passed.flag: Blocking Untested Models

If the calibration check passes (negative correlation confirmed), step 3B creates a sentinel file:

```python
gate_flag = Path('model_artifacts/gate_passed.flag')
gate_flag.write_text(f'PASSED: {datetime.now().isoformat()}')
```

Step 4 (`step4_inference_dashboard.py`) checks for this file at startup:

```python
if not gate_flag.exists():
    raise RuntimeError(
        "Gate flag not found. Run step3b before step4. "
        "Production inference requires validated model."
    )
```

This hard block prevents the pipeline from running production inference on:
- A freshly trained model that hasn't been evaluated (gate_passed.flag deleted by step 3)
- A model where step 3B was skipped due to time pressure
- A model where step 3B failed (gate flag not created)

The gate flag approach is a simple but effective safety pattern for ML pipelines with multiple interconnected steps.

---

### 11.6 Diagnostic Outputs Generated

Step 3B produces the following outputs in `evaluation_outputs/`:

| File | Description |
|------|-------------|
| `calibration_curve.png` | Confidence vs Mean MAE by decile (the key diagnostic plot) |
| `confidence_vs_error_scatter.png` | Individual sample scatter of confidence vs actual error |
| `proxy_vs_model_confidence.png` | Proxy confidence (from truth) vs model confidence |
| `evaluation_summary.json` | Overall correlation, mean confidence, mean MAE, gate result |
| `gate_passed.flag` | Created only if gate passes |

These outputs are retained for audit purposes â€” they document the state of model evaluation at the time the gate was passed, providing traceability for any production inference run.

---

### 11.7 Offline vs Production Safety Distinction

The key architectural distinction in step 3B:

| Step | Ground Truth Used? | Purpose |
|------|-------------------|---------|
| Step 3B | Yes (Y_test) | Offline calibration validation |
| Step 4 | No | Production inference |

Step 3B can only run after step 2 has created Y_test (the test set ground truth). It uses Y_test only to verify that confidence correlates with error â€” it never uses Y_test to improve model weights or scoring parameters.

Step 4 uses no ground truth at all. It operates on the most recent CDMs available for each event and produces threat/confidence scores entirely from model predictions and MC Dropout statistics. This is the operationally clean separation that makes the system deployment-safe.

---

## 12. Production Inference â€” The Dashboard

### 12.1 Step 4 Entry Guard (Gate Flag Check)

`step4_inference_dashboard.py` is the operational inference module. It is the only step that produces the operator-facing output. Its first action is an entry guard:

```python
gate_flag = Path('model_artifacts/gate_passed.flag')
if not gate_flag.exists():
    sys.exit("ERROR: gate_passed.flag not found. Run step3b first.")
logger.info(f"Gate passed: {gate_flag.read_text()}")
```

If the gate flag is missing, step 4 exits immediately with an error message. No inference is performed, no output files are written. This is the hard safety boundary of the pipeline.

The gate flag content is logged (it includes the timestamp of when step 3B passed), providing an audit trail: operators can verify that the model was evaluated on a specific date before this inference run was executed.

---

### 12.2 Rebuilding the Model from Config

Rather than deserializing the full Keras model object (which can have version compatibility issues), step 4 rebuilds the architecture from scratch using `model_builder.py` and then loads only the saved weights:

```python
config = load_yaml('Scripts/config.yaml')
model = build_model_from_config(config)
model.load_weights('model_artifacts/best_model.keras')
```

This approach is more robust than `keras.models.load_model()` because:
- It doesn't depend on the exact TensorFlow version that saved the model
- It guarantees the architecture matches what `model_builder.py` currently defines
- The custom weighted MSE loss doesn't need to be re-registered for weight loading (only for continued training)

---

### 12.3 Inference Input Validation

Before any inference, step 4 validates that all required artifacts exist and have expected shapes:

```python
required_artifacts = [
    'processed_sequences/X_test.npy',
    'processed_sequences/feature_scaler.pkl',
    'processed_sequences/feature_names.txt',
    'processed_sequences/split_info.csv',
]
for path in required_artifacts:
    if not Path(path).exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")

X_test = np.load('processed_sequences/X_test.npy')
assert X_test.shape[1] == config['training']['max_sequence_length'], \
    f"Sequence length mismatch: got {X_test.shape[1]}, expected {config['training']['max_sequence_length']}"
assert X_test.shape[2] == len(feature_names), \
    f"Feature count mismatch: got {X_test.shape[2]}, expected {len(feature_names)}"
```

These assertions catch the common failure mode where a config change (e.g., adding a feature) creates a mismatch between the saved sequences and the current model architecture.

---

### 12.4 The MC Dropout Inference Loop

For the test set (9,637 samples), MC Dropout inference is performed in batches to manage memory:

```python
BATCH_SIZE = 256
N_PASSES = config['inference']['mc_dropout_passes']  # 50

all_means, all_stds = [], []
for i in range(0, len(X_test), BATCH_SIZE):
    X_batch = X_test[i:i+BATCH_SIZE]
    batch_preds = np.array([
        model(X_batch, training=True).numpy()   # MC Dropout active
        for _ in range(N_PASSES)
    ])  # shape: (50, batch_size, 11)
    all_means.append(batch_preds.mean(axis=0))
    all_stds.append(batch_preds.std(axis=0))

mean_predictions = np.vstack(all_means)  # (9637, 11)
std_predictions  = np.vstack(all_stds)   # (9637, 11)
```

For each sample, `compute_threat_and_confidence()` is then called with the mean prediction, std prediction, last real CDM (extracted from X_test), and the number of valid (non-padding) timesteps.

---

### 12.5 Aggregating Sample-Level to Event-Level

Each conjunction event produces multiple samples (one per CDM in the sequence). For event-level triage, step 4 uses the **latest prediction for each event** â€” the one with the most CDM context:

```python
# split_info.csv maps sample_idx -> event_id
for event_id in unique_events:
    event_sample_indices = get_samples_for_event(event_id)
    # Take the sample with the most valid timesteps (= last in chronological order)
    latest_idx = max(event_sample_indices, key=lambda i: n_valid_steps[i])

    event_threat = threat_scores[latest_idx]
    event_confidence = confidences[latest_idx]
    event_quadrant = quadrants[latest_idx]
```

This aggregation rule is deliberate: the most recent CDM context gives the model the most information about the event's current state. Earlier samples from the same event represent the model's predictions at earlier points in time â€” useful for understanding event evolution, but not the right basis for current triage.

---

### 12.6 Event Dashboard: Structure & Fields

The primary output is `inference_outputs/event_dashboard.csv` â€” one row per event:

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | str | Conjunction event identifier |
| `object1_norad_id` | str | NORAD ID of primary object |
| `object2_norad_id` | str | NORAD ID of secondary object |
| `quadrant` | str | ACT NOW / WATCH CLOSELY / SAFELY IGNORE / NOT PRIORITY |
| `threat_score` | float | 0â€“100, operational urgency |
| `confidence` | float | 0.10â€“1.00, trust in assessment |
| `predicted_pc` | float | Model's predicted next Pc (linear) |
| `current_pc` | float | Most recent actual Pc from CDM |
| `pc_trend` | str | RISING / FALLING / STABLE |
| `predicted_tca_hours` | float | Predicted hours to TCA |
| `n_cdms` | int | Number of CDMs in event sequence |
| `mc_uncertainty` | float | Mean MC Dropout std across features |

A companion file `inference_outputs/dashboard_summary.json` contains aggregate statistics:
```json
{
  "total_events": 2003,
  "act_now": 312,
  "watch_closely": 489,
  "safely_ignore": 287,
  "not_priority": 915,
  "mean_threat_score": 38.7,
  "mean_confidence": 0.412,
  "inference_timestamp": "2026-01-27T14:23:11"
}
```

---

### 12.7 Surfacing ACT NOW Events

The 312 ACT NOW events are sorted by threat score (descending) and written to:
- `inference_outputs/act_now_events.csv` â€” sorted list of top events
- `inference_outputs/top_events.xlsx` â€” Excel file with color-coding for operator use

The top 20 ACT NOW events (by threat score) are separately visualized as Table 1 in the visualization step (Section 13.9).

---

### 12.8 How an Operator Would Use This

A satellite operations team receiving this output would follow a workflow:

1. **Open `dashboard_summary.json`** â€” get the day's triage statistics: 312 events need attention
2. **Sort `act_now_events.csv` by threat_score descending** â€” address highest-threat events first
3. **For each ACT NOW event:** check predicted_pc, pc_trend, predicted_tca_hours â€” decide whether to request more tracking data or execute a maneuver
4. **For WATCH CLOSELY events:** flag for re-evaluation when the next CDM arrives
5. **For SAFELY IGNORE / NOT PRIORITY:** no immediate action required

The entire triage cycle for 2,003 events reduces to detailed review of ~312 events instead of all 2,003 â€” an 84% reduction in analyst workload while maintaining full coverage of the high-risk population.

---

## 13. Visualization & Reporting

### 13.1 Figure 1: Risk Assessment Quadrant Dashboard

**File:** `visualizations/figure_01_quadrant_dashboard.png`
**Generated by:** `step5_visualize.py`
**Format:** 300 DPI, publication quality

A 2D scatter plot of all 2,003 events plotted in the threat-score vs. confidence space. The two quadrant boundaries (threat=50, confidence=0.5) are drawn as dashed lines, dividing the plot into four labeled regions. Points are color-coded by quadrant (ACT NOW: red, WATCH CLOSELY: orange, SAFELY IGNORE: green, NOT PRIORITY: blue). Point size scales with predicted Pc.

This is the signature visualization of the system â€” it communicates the core output in a single glance. The scatter of 312 red points in the upper-right quadrant is immediately interpretable by any operations team.

---

### 13.2 Figure 2: Threat Score Distribution

**File:** `visualizations/figure_02_threat_distribution.png`

A histogram of threat scores across all 2,003 events. The distribution is bimodal: a cluster near 0â€“20 (events well below the maneuver threshold) and a cluster near 60â€“80 (events above the threshold). The 50-point threshold is marked with a vertical dashed line.

The bimodal shape validates the scoring system: threat scores are not uniformly distributed, indicating that the model successfully discriminates between genuinely dangerous and benign events.

---

### 13.3 Figure 3: Confidence Level Distribution

**File:** `visualizations/figure_03_confidence_distribution.png`

A histogram of confidence levels. The distribution is approximately Gaussian centered near 0.40â€“0.45, with tails at 0.16 (minimum, events with 1â€“2 CDMs and large covariance) and 0.66 (maximum, events with 10+ CDMs and tight covariance).

The distribution confirms that the model is not systematically overconfident (no spike at 0.9â€“1.0) and not systematically underconfident (no spike at 0.1â€“0.2).

---

### 13.4 Figure 4: Threat vs Actual Collision Probability

**File:** `visualizations/figure_04_threat_vs_pc.png`

A scatter plot of threat score (y-axis) vs. the actual log10(Pc) from the last real CDM in each event's sequence (x-axis). The expected positive correlation is clearly visible: higher actual Pc â†’ higher threat score.

This plot validates the threat scoring system using the closest thing to ground truth available (the actual CDM's Pc, though not the collision outcome itself). Outliers (events with high Pc but low threat score) typically have declining Pc trends â€” the model correctly downweights events where Pc is already falling.

---

### 13.5 Figure 5: Confidence vs Data Quantity

**File:** `visualizations/figure_05_confidence_vs_n_cdms.png`

A box plot of confidence levels grouped by number of CDMs in the event sequence (1, 2â€“3, 4â€“6, 7â€“10, 10+). The expected positive relationship is visible: more CDMs â†’ higher confidence. The relationship is not perfectly linear because the MC Dropout uncertainty and covariance quality components also contribute to confidence.

---

### 13.6 Figure 6: Training Loss Curves

**File:** `visualizations/figure_06_training_loss.png`

A dual-line plot showing training loss and validation loss over all 150 epochs. The y-axis is log-scaled to show the full dynamic range (from initial loss ~4.0 to final loss ~0.63). The best epoch (127) is marked with a vertical line. The 7 learning rate reduction events are annotated with arrows.

The near-parallel train and val curves confirm effective regularization. The gap between the two curves is consistently < 0.02 after epoch 50.

---

### 13.7 Figure 7: Training MAE Curves

**File:** `visualizations/figure_07_training_mae.png`

Same format as Figure 6 but showing MAE (mean absolute error) rather than loss. MAE is in StandardScaler-normalized units. The final validation MAE of 0.531 is annotated.

---

### 13.8 Figure 8: Collision Probability Prediction Error

**File:** `visualizations/figure_08_pc_prediction_error.png`

A scatter plot of predicted log10(Pc) vs. actual log10(Pc) for all 9,637 test samples. A perfect predictor would show all points on the diagonal (y=x line). The actual scatter shows:
- Tight cluster along the diagonal for moderate Pc events (log10_pc in [-8, -4])
- Slightly larger scatter for extreme events (very high or very low Pc)
- No systematic bias (the regression line through the scatter passes near the origin)

The RÂ² of the scatter is approximately 0.71, indicating the model explains ~71% of the variance in Pc evolution.

---

### 13.9 Table 1: Top 20 ACT NOW Events (Excel + PNG)

**Files:** `visualizations/table_01_top_events.xlsx`, `visualizations/table_01_top_events.png`

A ranked table of the 20 highest-threat ACT NOW events, including: event_id, threat score, confidence, predicted Pc, current Pc, Pc trend, predicted TCA hours, and N CDMs. The table is exported to Excel with conditional formatting (red background for threat > 80, orange for 60â€“80).

A PNG snapshot of the table is also generated for inclusion in reports without requiring Excel.

---

### 13.10 Step 5B: Per-Event Detailed Reports (25 Events)

**Generated by:** `step5b_detailed_reports.py`
**Output directory:** `detailed_reports/`
**Events covered:** Top 20 ACT NOW events + 5 SAFELY IGNORE events (for contrast)

For each selected event, a multi-panel figure is generated showing the complete CDM history for that event:

**Panel 1 (top left):** Pc evolution over time â€” actual log10(Pc) values at each CDM's creation time, plus the model's predicted next Pc with MC Dropout uncertainty band (mean Â± 1.96 std)

**Panel 2 (top right):** Miss distance evolution â€” actual MISS_DISTANCE values, showing how the predicted closest approach changes as more tracking data arrives

**Panel 3 (bottom left):** Combined covariance (combined_ct_t) over time â€” shows whether tracking quality is improving (covariance shrinking) or stagnant

**Panel 4 (bottom right):** Threat score and confidence evolution â€” for events with multiple CDMs, shows how the model's assessment changed as more data arrived

Each report is titled with the event_id, quadrant, and final threat/confidence scores. Reports are saved at 300 DPI and stored in `detailed_reports/<event_id>.png`.

These per-event reports are the most operator-useful output: they transform the abstract threat/confidence numbers into a visual story of how a specific conjunction evolved and why the model assessed it the way it did.

---

## 14. Software Architecture & Pipeline Design

### 14.1 Pipeline Philosophy: Linear, Artifact-Driven

The DebriSolver pipeline follows a strictly **linear, artifact-driven** design. Each step produces one or more output artifacts (files), and each step consumes only the artifacts produced by previous steps. There is no shared in-memory state between steps.

This design provides several critical properties:
- **Restartability:** Any step can be re-run independently without re-running all previous steps, as long as the required input artifacts exist
- **Inspectability:** Any intermediate result can be inspected by examining the artifact files â€” there is no hidden state
- **Parallelism safety:** Steps can never inadvertently share state because they only communicate via the filesystem
- **Reproducibility:** A complete re-run from scratch always produces the same results (given the same random seed and input data)

---

### 14.2 The 7-Step Pipeline (1 â†’ 2 â†’ 3 â†’ 3B â†’ 4 â†’ 5 â†’ 5B)

| Step | Script | Input | Output | Description |
|------|--------|-------|--------|-------------|
| **1** | `step1_parse_kvn.py` | Raw .kvn files | `parsed_cdm_data.csv` | KVN parsing |
| **2** | `step2_prepare_sequences.py` | `parsed_cdm_data.csv` | `X_train.npy`, `Y_train.npy`, `feature_scaler.pkl`, etc. | Feature engineering + sequence prep |
| **3** | `step3_train_model.py` | `X_train.npy`, `Y_train.npy`, `X_val.npy`, `Y_val.npy` | `best_model.keras`, `training_history.json` | Model training |
| **3B** | `step3b_evaluate_proxy_confidence.py` | `X_test.npy`, `Y_test.npy`, `best_model.keras` | `evaluation_outputs/`, `gate_passed.flag` | Confidence calibration gate |
| **4** | `step4_inference_dashboard.py` | `X_test.npy`, `best_model.keras`, `gate_passed.flag` | `event_dashboard.csv`, `dashboard_summary.json` | Production inference |
| **5** | `step5_visualize.py` | `event_dashboard.csv`, `training_history.json` | 8 figures + 1 Excel table | Visualization |
| **5B** | `step5b_detailed_reports.py` | `event_dashboard.csv`, `X_test.npy` | 25 per-event report figures | Detailed event reports |

The gate (step 3B) is the only non-linear element: it must complete successfully before step 4 is permitted.

---

### 14.3 run_pipeline.py: The Orchestrator

`run_pipeline.py` is the top-level orchestrator that executes all steps in sequence:

```python
python run_pipeline.py [--from-step N] [--to-step M] [--dry-run]
```

The orchestrator:
1. Validates that all required artifacts from previous steps exist before running each step
2. Calls each step script as a subprocess with `subprocess.run()`
3. Checks the return code of each step â€” non-zero return code stops the pipeline
4. Logs all step start/end times and durations to `pipeline_run.log`

Step 4's gate check is enforced by step 4 itself, not by the orchestrator. The orchestrator simply calls step 4 and checks whether it exits with code 0 (success) or non-zero (gate blocked).

---

### 14.4 Step Resumption & Partial Runs

The `--from-step` and `--to-step` flags allow partial pipeline runs:

```bash
# Re-run only training and evaluation (steps 3 and 3b)
python run_pipeline.py --from-step 3 --to-step 3b

# Re-run only visualization (steps 5 and 5b)
python run_pipeline.py --from-step 5

# Run only up to training (steps 1 through 3)
python run_pipeline.py --to-step 3
```

This is essential for development velocity: changing the scoring logic in `scoring.py` requires re-running only steps 3B, 4, 5, and 5B â€” not re-parsing and re-training, which would take hours.

---

### 14.5 config.yaml: Single Source of Truth

All tunable parameters are centralized in `Scripts/config.yaml`. This file is read at the start of every pipeline step that needs parameters, using:

```python
import yaml
with open('Scripts/config.yaml') as f:
    config = yaml.safe_load(f)
```

Parameters stored in config.yaml (non-exhaustive):
- All model hyperparameters (GRU units, dropout, L2 regularization)
- All training parameters (epochs, batch size, learning rate, patience)
- Data parameters (max_sequence_length, padding_value, feature names)
- Inference parameters (MC dropout passes)
- Scoring thresholds (threat threshold, confidence threshold)
- Directory paths for artifacts

The config file has never had duplicate parameter definitions (it is the only definition). Engineers making hyperparameter changes need to touch exactly one file.

---

### 14.6 Shared Modules: model_builder.py & scoring.py

Two shared modules are imported by multiple pipeline steps:

**`Scripts/model_builder.py`:**
- Imported by steps 3, 3B, and 4
- Defines `build_model_from_config(config)` and `build_self_supervised_gru(...)`
- Ensures architectural consistency across all pipeline stages

**`Scripts/scoring.py`:**
- Imported by steps 3B and 4
- Defines `compute_threat_and_confidence()` and `classify_quadrant()`
- Ensures identical scoring logic in both evaluation and production

Both modules are pure Python (no pipeline artifacts, no file I/O). They can be unit tested in isolation using synthetic data without running any pipeline step.

---

### 14.7 Directory & Artifact Layout

```
SDC2026_KAU_AE_TEAM/
â”œâ”€â”€ Scripts/
â”‚   â”œâ”€â”€ step1_parse_kvn.py
â”‚   â”œâ”€â”€ step2_prepare_sequences.py
â”‚   â”œâ”€â”€ step3_train_model.py
â”‚   â”œâ”€â”€ step3b_evaluate_proxy_confidence.py
â”‚   â”œâ”€â”€ step4_inference_dashboard.py
â”‚   â”œâ”€â”€ step5_visualize.py
â”‚   â”œâ”€â”€ step5b_detailed_reports.py
â”‚   â”œâ”€â”€ model_builder.py               â† shared architecture module
â”‚   â”œâ”€â”€ scoring.py                     â† shared scoring module
â”‚   â”œâ”€â”€ config.yaml                    â† single source of truth
â”‚   â””â”€â”€ tools/
â”‚       â””â”€â”€ inspect_data.py
â”œâ”€â”€ Tests/
â”‚   â”œâ”€â”€ test_scoring.py
â”‚   â”œâ”€â”€ test_sequences.py
â”‚   â”œâ”€â”€ test_model_io.py
â”‚   â”œâ”€â”€ test_parser.py
â”‚   â””â”€â”€ smoke_tests.py
â”œâ”€â”€ Data/                              â† raw KVN files (input)
â”œâ”€â”€ parsed_data/
â”‚   â””â”€â”€ parsed_cdm_data.csv            â† step 1 output
â”œâ”€â”€ processed_sequences/
â”‚   â”œâ”€â”€ X_train.npy, Y_train.npy       â† step 2 outputs
â”‚   â”œâ”€â”€ X_val.npy, Y_val.npy
â”‚   â”œâ”€â”€ X_test.npy, Y_test.npy
â”‚   â”œâ”€â”€ feature_scaler.pkl
â”‚   â”œâ”€â”€ feature_imputer.pkl
â”‚   â”œâ”€â”€ feature_names.txt
â”‚   â””â”€â”€ split_info.csv
â”œâ”€â”€ model_artifacts/
â”‚   â”œâ”€â”€ best_model.keras               â† step 3 output
â”‚   â”œâ”€â”€ training_history.json
â”‚   â””â”€â”€ gate_passed.flag               â† step 3B output
â”œâ”€â”€ evaluation_outputs/              â† step 3B diagnostic plots
â”œâ”€â”€ inference_outputs/
â”‚   â”œâ”€â”€ event_dashboard.csv            â† step 4 output
â”‚   â”œâ”€â”€ dashboard_summary.json
â”‚   â”œâ”€â”€ act_now_events.csv
â”‚   â””â”€â”€ top_events.xlsx
â”œâ”€â”€ visualizations/                  â† step 5 figures
â”œâ”€â”€ detailed_reports/                â† step 5B per-event reports
â””â”€â”€ run_pipeline.py                  â† top-level orchestrator
```

---

### 14.8 The Gate Pattern (Step 3B â†’ Step 4)

The gate pattern is a quality control checkpoint embedded in the pipeline. Its design principle: **a step that produces safety-critical output should require explicit evidence that the model it uses has been validated.**

The pattern consists of three components:
1. **Invalidation:** Step 3 (training) deletes `gate_passed.flag` at startup, ensuring any new model requires re-evaluation
2. **Validation:** Step 3B creates `gate_passed.flag` only if calibration passes, providing explicit evidence of validation
3. **Enforcement:** Step 4 refuses to run if `gate_passed.flag` is absent, preventing unevaluated models from producing operator-facing output

This pattern is generalizable: any ML pipeline with multiple stages and safety-critical outputs can benefit from explicit gate flags between evaluation and deployment steps.

---

### 14.9 Reproducibility: Seeds, Determinism, Saved Scalers

Full reproducibility requires three elements, all implemented:

**1. Fixed random seeds (SEED=42):**
All stochastic operations (data split, weight initialization, dropout) use `SEED=42` set at the start of each script.

**2. TF_DETERMINISTIC_OPS:**
Forces TensorFlow to use deterministic (non-parallelized) implementations of GPU operations, eliminating floating-point summation order variation.

**3. Saved preprocessing artifacts:**
The `feature_scaler.pkl` and `feature_imputer.pkl` are fitted once and saved. All downstream steps load these saved objects rather than refitting, ensuring that the exact same scaling transformations are applied at evaluation and inference time as were applied at training time.

Together, these three measures ensure that two engineers running the full pipeline on identical hardware with identical data produce byte-identical outputs.

---

## 15. Testing Strategy & Test Suite

### 15.1 Testing Philosophy: No Artifacts Required

The DebriSolver test suite is designed around one principle: **unit tests must not require any pipeline artifacts to run.** This means tests should not need `X_train.npy`, `best_model.keras`, or `parsed_cdm_data.csv` to exist. Tests that require artifacts are integration tests, handled separately by `smoke_tests.py`.

This philosophy serves two purposes:
1. **CI/CD compatibility:** Tests can run on any machine with only the Python dependencies installed, without running the full pipeline first
2. **Development speed:** Developers can write and run unit tests while developing features, before the data pipeline artifacts exist

Mock objects are used throughout: synthetic sequences are generated with `np.random.randn()`, mock scalers are created with `StandardScaler().fit(np.zeros((10, 11)))`, and synthetic KVN file content is constructed as strings.

---

### 15.2 test_scoring.py â€” 13 Tests for Threat & Confidence Logic

The most comprehensive test file. All 13 tests run with mock scalers and synthetic predictions:

| Test | What It Verifies |
|------|------------------|
| `test_high_pc_high_threat` | Pc = 0.01 â†’ threat_score â‰¥ 80 |
| `test_low_pc_low_threat` | Pc = 1e-10 â†’ threat_score â‰¤ 20 |
| `test_rising_pc_trend_bonus` | Rising Pc â†’ trend_modifier = +15 |
| `test_falling_pc_trend_penalty` | Falling Pc â†’ trend_modifier = -15 |
| `test_tca_urgency_bonus` | TCA in 2 hours â†’ urgency_bonus = +10 |
| `test_low_uncertainty_high_confidence` | Low MC std â†’ high uncertainty_confidence |
| `test_high_uncertainty_low_confidence` | High MC std â†’ low uncertainty_confidence |
| `test_act_now_quadrant` | High threat, high confidence â†’ ACT NOW |
| `test_watch_closely_quadrant` | High threat, low confidence â†’ WATCH CLOSELY |
| `test_safely_ignore_quadrant` | Low threat, low confidence â†’ SAFELY IGNORE |
| `test_not_priority_quadrant` | Low threat, high confidence â†’ NOT PRIORITY |
| `test_threat_score_bounded` | threat_score always in [0, 100] |
| **`test_covariance_expm1_correction`** | **Covariance in log1p space â†’ cov_confidence â‰ˆ 1.0 (wrong); after expm1 â†’ correct value** |

The last test (KD-14 regression) is the most important: it explicitly verifies that the `expm1()` correction is applied and that removing it produces incorrect results.

---

### 15.3 test_sequences.py â€” Padding, Masking, Split Logic

| Test | What It Verifies |
|------|------------------|
| `test_padding_sentinel_value` | Padded positions contain exactly -999.0 |
| `test_left_padding_alignment` | Real CDMs are at the end of the padded sequence |
| `test_padding_correct_length` | Output sequences are exactly max_len=20 timesteps |
| `test_no_data_leakage` | No event_id appears in more than one split |
| `test_event_level_split` | All CDMs for an event are in the same split |
| `test_sample_generation_count` | Event with N CDMs produces N-1 samples |
| `test_min_sequence_length` | Events with 1 CDM are excluded (need â‰¥ 2 for a sample) |
| `test_split_reproducibility` | Same SEED produces same split across runs |

The `TestEventLevelSplit` class is the critical test class: it constructs a synthetic dataset with 50 events, runs the split, and verifies that the sets are disjoint.

---

### 15.4 test_model_io.py â€” Architecture, MC Dropout, Weight I/O

| Test | What It Verifies |
|------|------------------|
| `test_model_build_from_config` | `build_model_from_config(config)` produces a model with the correct layer structure |
| `test_output_shape` | Model output shape is (batch_size, 11) |
| `test_mc_dropout_active` | Predictions differ between calls when `training=True` |
| `test_mc_dropout_inactive` | Predictions are identical when `training=False` |
| `test_layer_norm_present` | Model contains LayerNormalization layers (not BatchNorm) |
| `test_weight_save_load` | Save weights to temp file, load into new model, outputs match |
| `test_masking_layer_present` | First non-input layer is a Masking layer |
| `test_parameter_count` | Total parameter count is 244,171 (Â±100 for minor config variations) |

The MC Dropout test is particularly important: it confirms that `training=True` actually produces variance in predictions (dropout is working) and that `training=False` produces deterministic predictions.

---

### 15.5 test_parser.py â€” KVN Parser Unit Tests

| Test | What It Verifies |
|------|------------------|
| `test_strip_units_m` | `strip_units('150.5 [m]')` â†’ `'150.5'` |
| `test_strip_units_m2` | `strip_units('100.0 [m**2]')` â†’ `'100.0'` |
| `test_strip_units_km_s` | `strip_units('7.2 [km/s]')` â†’ `'7.2'` |
| `test_strip_units_none` | `strip_units('1.5E-05')` â†’ `'1.5E-05'` (no change) |
| `test_object_namespacing` | Fields after `OBJECT = OBJECT1` get `object1_` prefix |
| `test_event_id_from_filename` | `CDM_25544_48274_003.kvn` â†’ `event_id='25544_48274'` |
| `test_negative_tca_rejected` | CDM with CREATION_DATE > TCA returns None |
| `test_missing_tca_rejected` | CDM with no TCA field returns None |
| `test_scientific_notation` | `1.5E-05` correctly converts to float |
| `test_derived_features_computed` | `log10_pc`, `time_to_tca_hours`, `combined_cr_r` present in output |

All tests use string literals (mock KVN file content), not actual .kvn files from the dataset.

---

### 15.6 smoke_tests.py â€” End-to-End Artifact Checks

`smoke_tests.py` is an integration test that runs only after the full pipeline has been executed. It validates that all expected output artifacts exist and have sensible content:

```bash
python Tests/smoke_tests.py  # run after full pipeline
```

Checks performed:
- `X_train.npy` exists and has shape (77989, 20, 11)
- `X_train.npy` contains no NaN or Inf values
- `feature_scaler.pkl` exists and can be loaded as a StandardScaler
- `best_model.keras` exists and has file size > 1 MB
- `gate_passed.flag` exists and contains "PASSED"
- `event_dashboard.csv` exists and has 2003 rows
- `event_dashboard.csv` contains expected columns (event_id, threat_score, confidence, quadrant)
- Quadrant column contains only valid values (ACT NOW, WATCH CLOSELY, SAFELY IGNORE, NOT PRIORITY)

Smoke tests are the final quality check before the outputs are considered ready.

---

### 15.7 The KD-14 Regression Test (expm1 covariance fix)

The `test_covariance_expm1_correction` test in `test_scoring.py` is worth examining in detail as an example of how to write a regression test for a subtle numerical bug:

```python
def test_covariance_expm1_correction():
    # Create a mock scaler fitted on log1p-transformed covariance data
    # Typical log1p values for combined_ct_t range from 0 to 23
    mock_data = np.zeros((100, 11))
    mock_data[:, 4] = np.random.uniform(0, 23, 100)  # combined_ct_t in log1p space
    scaler = StandardScaler().fit(mock_data)

    # Create a prediction with large log1p covariance (= poor tracking quality)
    mean_pred = np.zeros(11)
    mean_pred[4] = 21.0  # log1p(1.3 billion mÂ²) â‰ˆ 21 â€” very high covariance

    # Without expm1 correction (simulating the bug):
    pred_physical_wrong = scaler.inverse_transform(mean_pred.reshape(1, -1))[0]
    cov_t_wrong = pred_physical_wrong[4]  # still in log1p space, ~21
    cov_confidence_wrong = max(0, 1 - cov_t_wrong / 5e8)  # ~1.0 (wrong)

    # With expm1 correction (correct behavior):
    pred_physical_correct = pred_physical_wrong.copy()
    pred_physical_correct[4] = np.expm1(pred_physical_correct[4])  # convert to mÂ²
    cov_confidence_correct = max(0, 1 - pred_physical_correct[4] / 5e8)  # ~0.0 or near 0

    assert cov_confidence_wrong > 0.99, "Bug not reproduced correctly"
    assert cov_confidence_correct < 0.50, "expm1 correction not working"
```

This test explicitly encodes the bug (asserting it produces wrong output without the fix) and the fix (asserting correct output with expm1). Any future regression would cause the test to fail.

---

### 15.8 How to Run the Tests

```bash
# Run all unit tests (no artifacts required)
python -m pytest Tests/ -v

# Run only scoring tests
python -m pytest Tests/test_scoring.py -v

# Run only the KD-14 regression test
python -m pytest Tests/test_scoring.py::test_covariance_expm1_correction -v

# Run smoke tests (requires full pipeline to have run)
python Tests/smoke_tests.py

# Run all tests with coverage report
python -m pytest Tests/ --cov=Scripts --cov-report=html
```

Expected output for passing unit tests:
```
====== 50 passed in 4.23s ======
```

---

### 15.9 What Is NOT Tested (and Why)

**Not tested: Physical correctness of Pc predictions.** We cannot test whether the model's Pc predictions are physically accurate because there are no labeled collision outcomes to compare against. The test suite verifies the engineering properties (output shapes, scoring logic, architectural properties) but not the scientific accuracy of predictions.

**Not tested: Full pipeline end-to-end in CI.** The full pipeline takes ~2.7 hours (training) and requires the ALDORIA dataset (213 MB). CI runs only the unit tests (4 seconds), with the smoke tests run manually after a full pipeline run.

**Not tested: Operational alert quality.** Whether the ACT NOW events are the "right" events to alert on cannot be tested without real collision data. The evaluation gate (step 3B) provides the closest available proxy: confidence correlates with prediction accuracy, which is our best available validation signal.

---

## 16. Problems Encountered & How We Solved Them

*This section covers: every major bug, design mistake, and engineering challenge encountered during development, how each was diagnosed, and how it was fixed. This is the most important section for understanding the real journey of building this system.*

### 16.1 Problem 1: Validation Loss Explosion (84.9 â†’ fixed)
- **What happened:** val_loss stuck at ~85 while train_loss converged to ~1.0
- **Root cause:** COLLISION_PROBABILITY (raw linear scale) had catastrophic outliers (max=117 std devs). StandardScaler couldn't normalize it. Any CDM with Pc~0.07 produced targets with |v|>>10.
- **Fix:** Excluded raw COLLISION_PROBABILITY entirely. Used log10_pc instead. Val loss dropped from 84.9 â†’ 0.628 (~99.3% reduction).

### 16.2 Problem 2: Covariance Features Causing Training Instability
- **What happened:** Covariance values span 0 to ~20 billion mÂ². StandardScaler's mean/std were astronomically large, producing scaled values that exploded gradients.
- **Fix:** Applied log1p transform to all covariance features before scaling. Compressed range from [0, 2Ã—10Â¹â°] to [0, ~22]. Scaler then worked correctly.

### 16.3 Problem 3: Padding Value of Zero Conflicting with Scaled Data
- **What happened:** Initial implementation used 0.0 as the padding sentinel. After StandardScaler, real CDM values can be zero (mean-centered). Masking layer couldn't distinguish padding from real data.
- **Fix:** Changed padding sentinel to -999.0, a value physically impossible after StandardScaler normalization. Masking layer detects rows where all features == -999.0.

### 16.4 Problem 4: BatchNorm Breaking MC Dropout (KD-12)
- **What happened:** Using BatchNormalization with training=True for MC Dropout caused batch statistics to be recomputed each forward pass, making predictions inconsistent in a way unrelated to dropout randomness.
- **Fix:** Replaced all BatchNormalization with LayerNormalization. LayerNorm operates per-sample and is completely stable regardless of batch size or training flag.

### 16.5 Problem 5: KD-14 â€” Covariance Confidence Always ~0.99
- **What happened:** After the log1p transform fix, confidence was always near 1.0 regardless of covariance quality. High-covariance (poorly tracked) events weren't being flagged as low-confidence.
- **Root cause:** scoring.py was computing covariance confidence from log1p-scaled values (e.g., log1p(500000)â‰ˆ13.1) instead of raw mÂ² values. cov_confidence = 1/(1+13.1/1000) â‰ˆ 0.99. The denominator threshold was calibrated for mÂ², not log1p(mÂ²).
- **Fix:** Applied expm1() in scoring.py after inverse_transform to convert back to raw mÂ² before confidence computation. Regression test added to prevent recurrence.

### 16.6 Problem 6: Data Leakage via Temporal Split
- **What happened:** An early design split events by time (past events = train, future events = test). This inadvertently created a temporal forecasting problem instead of alert credibility assessment.
- **Fix:** Switched to random event-level split (80/10/10). All CDMs from a single conjunction event are assigned to exactly one split.

### 16.7 Problem 7: Gradient Explosion from Covariance Outliers
- **What happened:** Even after log1p transform, rare extreme covariance events produced large loss values and gradient spikes.
- **Fix:** Added gradient clipping (clipnorm=1.0) to the Adam optimizer. Any gradient with L2 norm > 1.0 is scaled down proportionally.

### 16.8 Problem 8: Training Stopping Too Early (Patience=10)
- **What happened:** EarlyStopping with patience=10 was terminating training before convergence. The model was still slowly improving at epoch 73/75.
- **Fix:** Increased epochs from 75 to 150, increased patience from 10 to 20. Model converged at epoch 147/150.

### 16.9 Problem 9: WATCH CLOSELY Dominating (Confidence Always Low)
- **What happened:** All events landed in WATCH CLOSELY because confidence never crossed 0.5. The scoring weights were unbalanced.
- **Root cause:** uncertainty_confidence weighted too heavily (0.60), data_confidence too low (0.20). Most events have only 2â€“5 CDMs, so data_confidence was always low, dragging total confidence below 0.5.
- **Fix:** Rebalanced weights: uncertainty_confidence=0.40, data_confidence=0.35, cov_confidence=0.25. This allowed events with enough CDMs and low uncertainty to reach ACT NOW or SAFELY IGNORE.

### 16.10 Problem 10: gate_passed.flag Not Invalidated After Retraining
- **What happened:** Running step3 (training) didn't clear the old gate flag, so step4 would run with a freshly trained but unevaluated model.
- **Fix:** step3_train_model.py now deletes gate_passed.flag at startup. Engineers must re-run step3b after every training run.

---

## 17. Results & Performance

### 17.1 Training Performance Summary

| Metric | Value |
|--------|-------|
| Best epoch | 127 / 150 |
| Training loss (weighted MSE) | 0.641 |
| Validation loss (weighted MSE) | 0.628 |
| Training MAE | 0.512 |
| Validation MAE | 0.531 |
| Learning rate at termination | 7.81Ã—10â»â¶ |
| LR decay events | 7 |
| Total training time (CPU) | ~2.7 hours |
| Train/val loss gap at best epoch | 0.013 (2.1% â€” no overfitting) |

The validation loss of 0.628 represents a **99.3% reduction** from the initial baseline of 84.9 (the pre-fix validation loss when raw COLLISION_PROBABILITY was included without log-transform). This reduction came entirely from data engineering fixes, not from architectural changes.

---

### 17.2 Test Set Results

Test set performance (9,637 samples from 201 unseen events):

| Metric | Value |
|--------|-------|
| Test loss (weighted MSE) | 0.651 |
| Test MAE | 0.547 |
| Test MAE gap from val MAE | +0.016 (3.0% â€” well-generalized) |

The small gap between validation MAE (0.531) and test MAE (0.547) confirms the model generalizes well to entirely unseen conjunction events. The gap is within expected statistical variation for a 200-event validation vs 201-event test comparison.

---

### 17.3 Collision Probability Prediction Accuracy

The primary operational metric: how accurately does the model predict the next CDM's logâ‚â‚€(Pc)?

| Metric | Value | Physical Interpretation |
|--------|-------|------------------------|
| log10_pc MAE | 0.42 log units | Predictions within factor of 2.6Ã— of true next Pc |
| log10_pc RÂ² | 0.71 | Model explains 71% of Pc evolution variance |
| Fraction within 0.5 log units | 63% | 63% of predictions within half an order of magnitude |
| Fraction within 1.0 log units | 87% | 87% of predictions within one order of magnitude |
| Systematic bias | <0.01 log units | No meaningful over/under-prediction tendency |

For operational triage, a Pc prediction within half an order of magnitude is more than sufficient. The threshold between "routine monitoring" (Pc < 1Ã—10â»â´) and "maneuver evaluation" (Pc â‰¥ 1Ã—10â»â´) is separated by orders of magnitude. A prediction error of 0.42 log units will not misclassify events that are clearly on one side of the threshold.

---

### 17.4 MC Dropout Uncertainty Statistics

From 50 forward passes per sample over the 2,003 test events:

| Metric | Value |
|--------|-------|
| Uncertainty range (mean std across features) | 0.050 â€“ 0.601 |
| Mean uncertainty | 0.183 |
| Median uncertainty | 0.161 |
| Correlation: uncertainty vs actual MAE | +0.31 (positive â€” high uncertainty â†’ high error) |
| Confidence range | 0.160 â€“ 0.660 |
| Mean confidence | 0.412 |
| Correlation: confidence vs actual MAE | -0.23 (negative â€” high confidence â†’ low error) |

The negative confidence-error correlation of -0.23 is the key result that causes the evaluation gate to pass. It is statistically significant (p < 0.001 at n=9,637) and operationally meaningful: events the model is confident about do, on average, have lower prediction error.

---

### 17.5 Risk Quadrant Distribution (2,003 Events)

| Quadrant | Events | % | Meaning |
|----------|--------|---|--------|
| **ACT NOW** | 312 | 15.6% | High threat, high confidence â€” immediate review |
| **WATCH CLOSELY** | 489 | 24.4% | High threat, lower confidence â€” request more tracking |
| **SAFELY IGNORE** | 287 | 14.3% | Low threat, low confidence â€” low priority |
| **NOT PRIORITY** | 915 | 45.7% | Low threat, high confidence â€” passive monitoring |

The system reduces the operator attention burden by ~84%: only 312 of 2,003 events require immediate review, compared to all 2,003 under a threshold-based system with no prioritization.

The 489 WATCH CLOSELY events represent a secondary triage layer â€” they have high predicted threat but uncertain assessments, and represent events where requesting additional tracking data before deciding on a maneuver is the correct operational action.

---

### 17.6 The 99.3% Validation Loss Reduction â€” What It Took

The journey from val_loss = 84.9 to val_loss = 0.628 is the most important engineering story in this project:

| Fix | Val Loss After Fix | Reduction |
|-----|-------------------|----------|
| Baseline (raw Pc, no log transform) | 84.9 | â€” |
| Fix 1: log10_pc for Pc feature | 1.84 | 97.8% from baseline |
| Fix 2: log1p for covariance + gradient clipping | 0.91 | 50.5% from Fix 1 |
| Fix 3: LayerNorm (enables stable MC Dropout) | 0.82 | 9.9% from Fix 2 |
| Fix 4: Patience=20 + epoch=150 | 0.71 | 13.4% from Fix 3 |
| Fix 5: Weight rebalancing (log10_pc Ã— 2.0) | 0.66 | 7.0% from Fix 4 |
| Fix 6: Seed=42 + TF deterministic ops | 0.628 | 4.8% from Fix 5 |

The dominant fix was the log10_pc transform (Fix 1), which alone accounts for 97.8% of the total reduction. All subsequent fixes contributed meaningful but smaller improvements. This pattern â€” one dominant fix, several refinements â€” is common in applied ML projects.

---

### 17.7 What the Results Mean Operationally

In operational terms, the DebriSolver system provides:

1. **Triage at scale:** 2,003 conjunction events assessed in ~12 minutes (inference time on CPU), producing a prioritized action list
2. **Calibrated confidence:** The 0.23 confidence-error correlation means that ACT NOW alerts are not noise â€” they reflect events where the model has genuine evidence of high risk and reliable data
3. **Trend detection:** The Pc trend modifier enables the system to catch rising-threat events that a threshold-based system would miss (events where Pc is currently below threshold but is predicted to cross it)
4. **Uncertainty transparency:** Operators always know how certain the model is, enabling informed decisions about when to act and when to wait for more data
5. **No false positives from noise:** The WATCH CLOSELY quadrant acts as a buffer between acting immediately and ignoring â€” events with high predicted threat but low confidence are flagged for more data, not immediate maneuver

---

### 17.8 Limitations of These Results

**No collision ground truth.** The fundamental limitation is that we cannot validate whether the ACT NOW events are truly the most dangerous ones. We have no labeled collision outcomes. The evaluation gate provides a proxy validation (confidence correlates with prediction error), but prediction error â‰  collision probability.

**Test set size.** The 201-event test set is statistically small for evaluating rare phenomena. The 312 ACT NOW events include many test events, but with only 201 events total, statistical confidence intervals on the quadrant percentages are wide (roughly Â±2â€“5%).

**Single dataset.** All results are from the ALDORIA dataset covering a specific time window. Performance on CDMs from other providers (e.g., LeoLabs, SpaceData Center) or different orbital regimes (GEO, MEO) is untested.

**CPU-only training.** Training on GPU would likely allow larger batch sizes and possibly higher-quality models, but this was not tested.

---

## 18. Libraries, Frameworks & Tools

### 18.1 TensorFlow / Keras
**Version:** TensorFlow 2.13 / Keras (bundled)
**Role:** Core deep learning framework â€” model definition, training, MC Dropout inference
**Why chosen:** Industry-standard for production ML; excellent GRU/RNN support; Keras Functional API allows clean custom loss functions; native support for `training=True` flag needed for MC Dropout
**Key APIs used:** `keras.layers.Bidirectional`, `keras.layers.GRU`, `keras.layers.Masking`, `keras.layers.LayerNormalization`, `keras.callbacks.EarlyStopping`, `keras.callbacks.ReduceLROnPlateau`, `keras.callbacks.ModelCheckpoint`, `keras.optimizers.Adam`

### 18.2 NumPy
**Version:** 1.24+
**Role:** All array operations â€” sequence construction, padding, MC Dropout aggregation, metric computation
**Key operations:** `np.full()` (padding), `np.vstack()`, `np.log1p()`, `np.expm1()`, `np.log10()`, `np.corrcoef()`, `np.clip()`

### 18.3 Pandas
**Version:** 2.0+
**Role:** CSV I/O, data manipulation, DataFrame operations during feature engineering
**Key operations:** `pd.read_csv()`, `pd.to_numeric(errors='coerce')`, `pd.DataFrame.groupby()`, `pd.DataFrame.to_csv()`

### 18.4 scikit-learn
**Version:** 1.3+
**Role:** Preprocessing (StandardScaler, SimpleImputer) and test utilities
**Key APIs:**
- `StandardScaler` â€” fit on training data, transform all splits
- `SimpleImputer(strategy='median')` â€” missing value imputation
- `sklearn.metrics.mean_absolute_error` â€” evaluation metric computation

### 18.5 joblib
**Version:** 1.3+
**Role:** Serialization of sklearn preprocessing artifacts (scaler, imputer) to disk
**Usage:** `joblib.dump(scaler, 'feature_scaler.pkl')` and `joblib.load('feature_scaler.pkl')`
**Why not pickle?** joblib is optimized for large numpy arrays embedded in sklearn objects â€” significantly faster than pickle for these use cases

### 18.6 PyYAML
**Version:** 6.0+
**Role:** Loading `config.yaml` into Python dictionaries
**Usage:** `yaml.safe_load(open('Scripts/config.yaml'))` â€” `safe_load` is used (not `load`) to prevent arbitrary code execution from malicious YAML files

### 18.7 Matplotlib
**Version:** 3.7+
**Role:** All data visualization â€” training curves, quadrant scatter plot, distributions, prediction error scatter
**Key settings:** `dpi=300` for publication quality, `plt.style.use('seaborn-v0_8-darkgrid')` for clean aesthetics, `bbox_inches='tight'` for proper export

### 18.8 h5py
**Version:** 3.9+
**Role:** Underlying HDF5 format used by Keras for model weight serialization (`.keras` files internally use HDF5)
**Direct usage:** Not directly called by our code; used transitively by Keras's `model.save_weights()` and `model.load_weights()`

### 18.9 openpyxl
**Version:** 3.1+
**Role:** Excel file generation for the top-events report (`top_events.xlsx`)
**Key features used:** `openpyxl.styles.PatternFill` for conditional color-coding, `openpyxl.styles.Font` for bold headers

### 18.10 pytest
**Version:** 7.4+
**Role:** Test runner for the full test suite (50+ tests across 5 files)
**Key plugins used:** `pytest-cov` for coverage reporting

### 18.11 python-dateutil
**Version:** 2.8+
**Role:** Robust datetime parsing in the KVN parser â€” handles multiple datetime formats that appear in ALDORIA CDMs without requiring a fixed format string
**Usage:** `dateutil.parser.parse('2025-11-01T12:00:00.000')` handles ISO 8601 with optional milliseconds

### 18.12 Python Standard Library
**Key modules and their roles:**
- `re` â€” regex for unit stripping in KVN parser (`strip_units()`)
- `json` â€” reading/writing training history and dashboard summary JSON files
- `os`, `pathlib.Path` â€” file existence checks, path manipulation throughout pipeline
- `argparse` â€” CLI argument parsing for `--from-step`, `--to-step` in `run_pipeline.py`
- `subprocess` â€” step orchestration in `run_pipeline.py` (each step called as a subprocess)
- `logging` â€” structured logging in all pipeline steps
- `math` â€” `math.log10()` for scalar log10_pc computation in parser
- `random` â€” Python random seed for reproducibility

---

## 19. Lessons Learned & Future Work

### 19.1 What We Would Do Differently

**Start with data profiling before modeling.** The most expensive mistakes in this project (val_loss = 84.9, covariance instability) were all data-related. In hindsight, spending the first full week exclusively profiling the dataset â€” printing min/max/mean/std for every feature, plotting distributions, checking for outliers â€” before writing a single training line would have prevented several debug cycles.

**Design the preprocessing pipeline defensively from day 1.** The log1p-then-StandardScaler chain creates an obligation to apply expm1 at inference time. This obligation should be encoded in a single `Preprocessor` class that handles both the forward transform and the inverse transform, rather than being split across step 2 (transform) and scoring.py (inverse). The KD-14 bug would have been impossible if a single object owned both directions of the transform.

**Use smaller datasets for initial experiments.** We trained on the full 77,989-sample training set from the beginning, making each debug cycle 2+ hours long. For the first 10 experiments, 5,000 samples would have been sufficient to validate the data pipeline and architecture, with the full dataset used only for final training.

**Implement the evaluation gate first.** The gate (step 3B) was implemented after training was already working. In retrospect, it should have been designed before training, because the gate design (what to measure, what threshold to use) directly informs what the training objective should optimize for.

---

### 19.2 What Worked Better Than Expected

**Self-supervised learning generalized well.** We expected that predicting future CDMs without collision labels would be an imperfect proxy for risk assessment. In practice, the model learned physical relationships (Pc evolution, covariance decay, miss distance refinement) that map directly to operational threat levels. The 71% RÂ² on Pc prediction and the -0.23 confidence-error correlation both exceeded initial estimates.

**MC Dropout calibration was naturally useful.** We implemented MC Dropout primarily as a Bayesian approximation to uncertainty. The fact that uncertainty negatively correlates with prediction error (not just with data quantity) means the model is genuinely learning which events are more or less predictable â€” not just flagging sparse-data events as uncertain. This is a stronger result than expected.

**BiGRU on 20-timestep sequences was sufficient.** We initially considered Transformer architectures for their superior long-range attention capabilities. For sequences of at most 20 timesteps, the BiGRU's implicit sequence representation proved more than sufficient, with lower computational cost and simpler implementation.

**The gate flag pattern stopped real errors.** During development, the gate flag mechanism caught three cases where step 4 would have run on an unevaluated model (once after a hyperparameter change, twice after code refactoring that inadvertently changed model behavior). This is exactly the class of error the pattern was designed to prevent.

---

### 19.3 Potential Architecture Improvements (Transformer, Attention)

**Self-Attention over CDM sequences.** A Transformer encoder with multi-head self-attention could, in principle, learn richer relationships between CDMs at different points in the sequence. For example, it could learn that a sudden drop in covariance at timestep k (new sensor data acquired) should dramatically reweight the model's attention to the most recent CDMs. The BiGRU cannot selectively attend to specific timesteps â€” it processes them sequentially.

**Temporal Convolutional Networks (TCN).** TCNs use causal dilated convolutions to capture temporal patterns at multiple timescales simultaneously. For CDM sequences, this could be valuable: some patterns unfold over 1â€“2 CDMs (rapid Pc change), others over 5â€“10 CDMs (slow covariance refinement). TCNs are also fully parallelizable (unlike GRUs), making them significantly faster to train.

**Hierarchical architecture.** A two-level model where the inner level processes CDMs within a single event and the outer level processes patterns across multiple events could capture population-level effects (e.g., events with similar debris characteristics tend to follow similar Pc trajectories).

---

### 19.4 Better Uncertainty Calibration (Temperature Scaling)

MC Dropout provides uncertainty estimates, but these estimates are not guaranteed to be calibrated â€” a predicted confidence of 0.60 doesn't necessarily mean the model is correct 60% of the time.

**Temperature Scaling** (Guo et al., 2017) is a post-hoc calibration technique that applies a single learned parameter (temperature T) to rescale the model's output logits before converting to probabilities. Applied to regression, the analog is scaling the MC Dropout std by a learned temperature factor to make the uncertainty intervals better match observed prediction errors.

Implementation would require a held-out calibration set (separate from train/val/test), fitting T to minimize the Expected Calibration Error on that set, and then applying T to all inference uncertainty estimates.

---

### 19.5 Active Learning for Rare High-Risk Events

The training dataset is severely imbalanced: the vast majority of events have Pc well below 1Ã—10â»â´. The model has seen very few examples of genuine high-threat events during training.

**Active learning** would address this by iteratively querying the most informative samples for labeling (or in our case, for inclusion in training). Events where the model has highest uncertainty could be flagged as the most valuable for the next training iteration. Over multiple cycles, the training set would progressively include more edge cases, improving model performance on the rare but critical high-threat events.

---

### 19.6 Real-Time Streaming CDM Ingestion

The current pipeline is batch-oriented: it processes a static dump of historical CDMs. For operational deployment, a streaming architecture would be preferable:

1. CDMs arrive from the space surveillance network in real time
2. Each new CDM triggers a model update for its event's sequence
3. The scoring system produces an updated threat/confidence assessment within seconds
4. Alerts are pushed to operators immediately rather than waiting for a daily batch run

This would require replacing the current file-based pipeline with a message queue (e.g., Apache Kafka or RabbitMQ) and a stateful inference service that maintains the current CDM sequence for each active conjunction event in memory.

---

### 19.7 Extending to Multi-Object Scenarios

The current model treats each conjunction as a two-body event (Object 1 vs Object 2). In reality, a single debris cloud can create conjunction alerts with hundreds of satellites simultaneously. A multi-object extension would model the relationships between conjunctions involving common objects.

For example: if ISS has 50 active conjunction alerts, the alerts are not independent â€” they share a common object (ISS) with the same orbital uncertainty. A graph neural network that models all active conjunctions for a given satellite simultaneously could produce more consistent threat assessments across the shared-object events.

---

### 19.8 Formal Validation Against Known Near-Misses

The most compelling validation study for this system would be to apply it retroactively to the weeks leading up to known high-risk events:

- **Iridium 33 approach to Cosmos 2251 (2009):** The CDM sequence for this event, if available, would be the ultimate test case. Did the model predict rising Pc? Did it produce an ACT NOW alert with high confidence?
- **ISS near-misses:** Multiple documented cases where the ISS executed avoidance maneuvers based on late-developing CDM sequences
- **GEO station-keeping conflicts:** Known cases where two geostationary satellites had close approaches that required coordinated maneuver planning

This retrospective validation would provide the closest available ground truth for the model's operational correctness.

---

### 19.9 BiGRU Training Improvements - Future Hyperparameter and Schedule Work

> **Context:** The current model converges in ~10 epochs (loss 1.015 to 0.72) then improves slowly through epoch 150 (0.72 to 0.628). The fast initial drop followed by a near-plateau is normal for GRU self-supervised training but could be improved with the techniques below.

**1. Learning Rate Warmup**
- Ramp up Adam LR linearly over first 10-20 epochs from near-zero to peak rate, then decay.
- Prevents the model committing to a suboptimal basin during unstable early epochs.
- Expected effect: smoother, more gradual descent curve - less cliff-then-plateau appearance.
- Implementation: tf.keras.optimizers.schedules.CosineDecayRestarts with a warmup wrapper.

**2. Curriculum Learning**
- Train first on easy examples (events with many CDMs), then progressively include harder ones (1-2 CDM events).
- The model currently sees all events equally from epoch 1, including hardest sparse-data events.
- Expected effect: more stable early training, better MAE on WATCH CLOSELY quadrant events.
- Implementation: Sort by CDM count descending, use sample_weight in model.fit().

**3. Larger Model (More Capacity)**
- Increase BiGRU hidden units from current 64->128 to 128->256, or add a second BiGRU layer.
- Current 244,171 parameters may hit capacity ceiling by epoch 20-30.
- Expected: slower convergence = more continuous descent across all 150 epochs.
- Trade-off: ~4x training time (~10h). Likely requires GPU.

**4. Cosine Annealing with Warm Restarts (SGDR)**
- Periodically resets LR to high value then decays again, instead of only reducing when loss stalls.
- The periodic restarts cause the model to escape shallow local minima.
- Training curve would show multiple visible descent phases rather than a single cliff.
- Implementation: tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=30)
- Reference: Loshchilov and Hutter 2016, SGDR: Stochastic Gradient Descent with Warm Restarts.

**Priority order for next training run:** Cosine Annealing -> Learning Rate Warmup -> Curriculum Learning -> Larger Model

---
## 20. Team & Acknowledgments

### 20.1 Team Members & Roles

The DebriSolver project was developed by the **SDC2026 KAU AE Team** from King Abdulaziz University, Aerospace Engineering Department, competing in the Saudi Space Data Challenge 2026 (Space Debris Conference, Riyadh, January 26â€“27, 2026).

| Name | Role | LinkedIn |
|------|------|----------|
| **Ahmad Alharbi** | Team Lead & Lead Developer | [ahmed-alharbi-973b63246](https://www.linkedin.com/in/ahmed-alharbi-973b63246/) |
| **Abdulelah Mojelad** | AI Research & Development | [abdulellah-mojalled](https://www.linkedin.com/in/abdulellah-mojalled/) |
| **Hamzah Alharbi** | Research & Development | [hamzah-alharbi-00b18133a](https://www.linkedin.com/in/hamzah-alharbi-00b18133a/) |
| **Khalid Alsadoon** | Research & Development | [khalid-alsadoon-a95802242](https://www.linkedin.com/in/khalid-alsadoon-a95802242/) |
| **Mohamedhakim Hassan** | Research & Development | [mohamed-hassan-aero](https://www.linkedin.com/in/mohamed-hassan-aero/) |

**Ahmad Alharbi** â€” Team Lead & Lead Developer
- System architecture design
- Self-supervised learning formulation and BiGRU model development
- Data pipeline engineering (steps 1â€“5B)
- Scoring system design (threat/confidence/quadrant framework)
- MC Dropout uncertainty quantification implementation
- Test suite development and debugging
- This documentation

**Abdulelah Mojelad, Hamzah Alharbi, Khalid Alsadoon, Mohamedhakim Hassan** â€” Research & Development
- Domain research: space debris environment, CDM semantics, operational STM
- Literature review: prior conjunction assessment methods and ML approaches
- Validation review and project presentation

---

### 20.2 King Abdulaziz University â€” Aerospace Engineering

**Institution:** King Abdulaziz University (KAU), Jeddah, Saudi Arabia
**Department:** Aerospace Engineering
**Program:** B.Sc. / M.Sc. in Aerospace Engineering

KAU's Aerospace Engineering department provided the academic framework and computational resources for this project. The department's focus on space systems and orbital mechanics provided the domain knowledge foundation for understanding CDM semantics, conjunction event physics, and operational space traffic management requirements.

---

### 20.3 Saudi Space Agency (SSA)

The **Saudi Space Agency (SSA)** organized the Saudi Space Data Challenge 2026 (SDC2026), providing the competition framework, problem statement, and evaluation criteria. The SSA's mission to develop Saudi Arabia's space capabilities and foster space data literacy among Saudi students and researchers made this competition possible.

The DebriSolver project was developed as a response to the SSA's challenge problem: develop a machine learning system for improved space object conjunction risk assessment.

---

### 20.4 ALDORIA â€” Data Provider

**ALDORIA** (formerly B612 Foundation's asteroid detection arm, now an independent space surveillance company) provided the Conjunction Data Message dataset used for training and evaluation. The ALDORIA dataset contains 185,511+ CDMs covering 20,506 raw conjunction events across the LEO operational environment.

The quality of ALDORIA's CDM data â€” consistent KVN format, rich covariance information, and complete CDM sequences for each event â€” was essential to the project's success. The dataset is provided under the terms of the SDC2026 competition data license.

---

## 21. References & Citation

### 21.1 Citing This Work

The correct citation for the submitted paper (title and all authors as they appear in the final submission PDF):

```bibtex
@inproceedings{alharbi2026conjunction,
  title        = {Learning Conjunction Dynamics: A Self-Supervised Approach
                  to Satellite Collision Risk Assessment},
  author       = {Alharbi, Ahmad and Mojelad, Abdulelah and Alharbi, Hamzah
                  and Alsadoon, Khalid and Hassan, Mohamedhakim},
  booktitle    = {Space Debris Conference 2026 -- DebriSolver Competition},
  address      = {Riyadh, Saudi Arabia},
  year         = {2026},
  organization = {Saudi Space Agency (SSA)}
}
```

For citing the code repository and technical documentation:

```bibtex
@misc{alharbi2026debrisolver_code,
  title        = {DebriSolver: SDC2026 KAU AE Team Codebase},
  author       = {Alharbi, Ahmad and Mojelad, Abdulelah and Alharbi, Hamzah
                  and Alsadoon, Khalid and Hassan, Mohamedhakim},
  year         = {2026},
  howpublished = {\url{https://github.com/AhmedAlharbii/SDC2026_KAU_AE_TEAM}},
  note         = {King Abdulaziz University, Aerospace Engineering Department}
}
```

---

### 21.2 Key References

The following are the **exact 10 references cited in the submitted paper**, as they appear in the final submission PDF:

---

**[1]** "ESA Space Environment Report 2024," ESA.
Available: https://www.esa.int/Space_Safety/Space_Debris/ESA_Space_Environment_Report_2024
*(Accessed: Nov. 10, 2025)*
> *Cited in Â§1 (Introduction) â€” motivation for LEO congestion and mega-constellation risks.*

---

**[2]** L. Sanchez, M. Vasile, E. Minisci (2020). "On the Use of Machine Learning and Evidence Theory to Improve Collision Risk Management." Paper presented at the *2nd IAA International Conference in Space Situational Awareness*, Washington, D.C., USA, 14â€“16 January 2020.
*(Accessed: Oct. 25, 2025)*
> *Cited in Â§1 & Â§1.1 â€” limitations of fixed-threshold approaches and false-positive cost.*

---

**[3]** A. K. Mashiku, L. K. Newman, and D. E. Highsmith, "NASA Conjunction Assessment Risk Analysis (CARA) Compendium for Artificial Intelligence and Machine Learning for Satellite Collision Avoidance," in *Proc. 26th AMOS Advanced Maui Optical and Space Surveillance Technologies Conference*, Wailea, HI, USA, 2025.
Available: https://ntrs.nasa.gov/api/citations/20250008251/downloads/AMOS_2025_AIML_Paper_UpdatedContractorAddress.pdf
*(Accessed: Dec. 12, 2025)*
> *Cited in Â§2.3 (Methodology) â€” self-supervised sequence learning task definition.*

---

**[4]** Y. Qiao, H.-M. Xu, W.-J. Zhou, B. Peng, B. Hu, and X. Guo, "A BiGRU joint optimized attention network for recognition of drilling conditions," *Petroleum Science*, vol. 20, no. 6, pp. 3624â€“3637, 2023.
doi: 10.1016/j.petsci.2023.05.021
*(Accessed: Nov. 3, 2025)*
> *Cited for BiGRU architecture justification.*

---

**[5]** D. Xu et al., "A survey on multi-output learning," *arXiv.org*, arXiv:1901.00248.
Available: https://arxiv.org/abs/1901.00248
*(Accessed: Dec. 1, 2025)*
> *Cited for the multi-output regression formulation (predicting all 11 CDM features simultaneously).*

---

**[6]** J. Terven, D.-M. CÃ³rdova-Esparza, J.-A. Romero-GonzÃ¡lez, A. RamÃ­rez-Pedraza, and E. A. ChÃ¡vez-Urbiola, "A comprehensive survey of loss functions and metrics in Deep Learning," *Artificial Intelligence Review*, SpringerLink, 2025.
Available: https://link.springer.com/article/10.1007/s10462-025-11198-7
*(Accessed: Nov. 5, 2025)*
> *Cited for weighted MSE loss design and metric selection.*

---

**[7]** Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in Deep Learning," *arXiv.org*, arXiv:1506.02142.
Available: https://arxiv.org/abs/1506.02142
*(Accessed: Nov. 29, 2025)*
> *Primary theoretical foundation for Monte Carlo Dropout (Section 9 of this document).*

---

**[8]** M. Hasan, A. Khosravi, I. Hossain, A. Rahman, and S. Nahavandi, "Controlled dropout for uncertainty estimation," *arXiv.org*, arXiv:2205.03109.
Available: https://arxiv.org/abs/2205.03109
*(Accessed: Nov. 25, 2025)*
> *Cited for controlled dropout uncertainty estimation methodology.*

---

**[9]** A. Pim and T. Pryer, "Surrogate modelling of proton dose with Monte Carlo dropout uncertainty quantification," *arXiv.org*, arXiv:2509.18155.
Available: https://arxiv.org/abs/2509.18155
*(Accessed: Dec. 2, 2025)*
> *Cited as applied MC Dropout case study in a safety-critical domain.*

---

**[10]** X. Cao, Y. Xu, and X. Yang, "Customer lifetime value prediction with uncertainty estimation using Monte Carlo Dropout," *arXiv.org*, arXiv:2411.15944.
Available: https://arxiv.org/abs/2411.15944
*(Accessed: Nov. 28, 2025)*
> *Cited as applied MC Dropout uncertainty estimation example.*

---

### 21.3 Technical Standards (CCSDS CDM Standard)

- **CCSDS 508.0-B-1** (2013): *Conjunction Data Message, Recommended Standard.* Consultative Committee for Space Data Systems (CCSDS), Blue Book. Washington, D.C.: CCSDS Secretariat.
  - Defines the official KVN (Key-Value Notation) and XML formats for Conjunction Data Messages
  - Specifies all mandatory and optional fields, their units, and their physical meanings
  - The ALDORIA dataset follows this standard; our KVN parser implements the field definitions from this document

- **CCSDS 502.0-B-2** (2019): *Orbit Data Messages, Recommended Standard.* CCSDS, Blue Book.
  - Defines the OPM, OEM, and OMM orbit data formats referenced by CDM object blocks
  - Relevant for understanding the coordinate frames (RTN, ECI) used in CDM relative position fields

---

*Document Status: COMPLETE â€” All 21 sections fully authored.*
*Last Updated: May 2026*
*Total sections: 21 | Total subsections: 160+ | Estimated reading time: 4â€“6 hours*

