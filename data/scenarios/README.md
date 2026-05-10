# SOAP Scenarios

This directory contains the nine fictional SOAP records authored *de novo* for this study, as described in §2.2 of the manuscript.

## File Naming Convention

Each scenario is stored as a separate plain-text file in UTF-8 encoding:

```
A1_acute_pain.txt
A2_dehydration.txt
A3_gas_exchange.txt
C1_self_management.txt
C2_activity_intolerance.txt
C3_chronic_pain.txt
P1_anxiety.txt
P2_caregiver_strain.txt
P3_spiritual_distress.txt
```

## Case Index

| Case ID | Category | NANDA-I diagnosis | Clinical setting | Body length (chars) |
|---|---|---|---|---|
| A1 | Acute | Acute pain (00132) | Post-laparoscopic appendectomy, postoperative day 1 | 717 |
| A2 | Acute | Deficient fluid volume (00027) | Geriatric heat-related dehydration | 803 |
| A3 | Acute | Impaired gas exchange (00030) | Chronic obstructive pulmonary disease, acute exacerbation | 854 |
| C1 | Chronic | Readiness for enhanced self-management (00162) | Diabetes mellitus, educational admission | 950 |
| C2 | Chronic | Activity intolerance (00092) | Heart failure (NYHA III), pre-discharge | 904 |
| C3 | Chronic | Chronic pain (00133) | Pre-total knee arthroplasty | 988 |
| P1 | Palliative | Anxiety (00146) | Pancreatic cancer, undergoing chemotherapy | 1,024 |
| P2 | Palliative | Caregiver role strain (00061) | Lung cancer, home-care transition | 1,162 |
| P3 | Palliative | Spiritual distress (00066) | Ovarian cancer, palliative care unit | 1,124 |

Mean body length: 947 Japanese characters (range 717–1,162).

## File Format

Each file begins with a metadata header followed by the four SOAP sections in Japanese:

```
case_id: A1
category: acute
nanda_diagnosis: 急性疼痛 (Acute Pain, 00132)
patient: 68歳男性、虫垂炎術後第1病日
scene: 手術翌朝の申し送り直前

S（主観的情報）
[patient's subjective narrative, including direct quotations]

O（客観的情報）
[vital signs, physical findings, laboratory values, observations]

A（アセスメント）
[nursing assessment with NANDA-I diagnosis and reasoning]

P（計画）
[nursing care plan, interventions, expected outcomes]
```

## Relationship to Fact Tags

Each scenario has 20 pre-specified clinical fact tags stored in `../tags/fact_tags.csv`. Tags are case-specific: tag `A1_01` belongs exclusively to case A1 and is evaluated only against the 180 SBARs generated for case A1 (30 trials × 2 temperatures × 3 models).

## Encoding

All scenario files are UTF-8 (no BOM). Verification:

```bash
file scenarios/*.txt
```

Should report each file as: `UTF-8 Unicode text`.

## Reuse and Adaptation

These scenarios are released under the MIT licence as part of this repository. Researchers wishing to:

1. **Replicate** the study with the same scenarios — use the files as provided.
2. **Adapt** the scenarios to other languages — translate the SOAP text and update the corresponding tag labels in `../tags/fact_tags.csv`.
3. **Extend** the scenario set — author additional SOAP records following the structure above and add the corresponding tags following the schema in `../tags/fact_tags.csv`.

## Note on Authorship and Validation

All nine scenarios were authored by the principal investigator without independent expert review prior to data collection. This is acknowledged as a limitation in §4.4 (limitation 6) of the manuscript. Independent groups using these scenarios are encouraged to apply their own validation procedures appropriate to their institutional context.
