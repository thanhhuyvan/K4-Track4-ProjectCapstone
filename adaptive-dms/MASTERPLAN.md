# Adaptive DMS Master Plan

Last updated: 2026-08-02

## 1. Mission

Build a lightweight, causal driver-monitoring system that:

- predicts `alert`, `drowsy`, `yawning`, `distracted`, and `microsleep`;
- runs in the car with a stable frozen global model;
- personalizes to one owner through a small cloud-trained adapter;
- keeps the global safety representation intact;
- produces the competition CSV format and a research-valid evaluation.

The final deployable is not one monolithic checkpoint:

```text
global_dms.pt       frozen shared model
owner_adapter.pt    small versioned owner update
owner_profile.json  prototypes, offsets, thresholds, calibration
```

## 2. Non-negotiable design rules

1. The inference path is causal. No backward fill or future frames.
2. Use LayerNorm, not BatchNorm, in trainable few-shot components.
3. Separate the competition lane from the research lane.
4. Split subjects and sessions before creating temporal windows.
5. Report 3D gaze angular error and 2D pitch/yaw MAE separately.
6. Do not infer `drowsy` or `microsleep` directly from raw DMD eye labels
   without an explicit temporal policy and validation.
7. Personalization begins with calibration and prototypes. More parameters are
   unlocked only when an owner holdout set proves they are needed.
8. Every owner update is versioned, validated, signed, and reversible.

## 3. Verified starting point

### Competition data

- Six labelled practice trips: 3,600 driver frames total.
- Ten scored trips: 18,000 driver frames with driver ground truth removed.
- Required driver states: five-class per-frame classification.
- Driver frames come from DMD source recordings and are released at 640x360.
- Practice clips may repeat source frames, so perception can run below 20 FPS
  while the state output is expanded back to every submitted frame.

### Partial DMD acquisition

- Subjects acquired: 1, 5, 6, 7, and 9.
- Five `s5` sessions, approximately 15 minutes total.
- 26,874 face-camera frames in the generated manifest.
- 23 annotated yawns and 128 annotated eye-closure events.
- Current material is useful for eye/yawn auxiliary tasks but contains no
  explicit five-class high-level driver-state target.
- The five acquired sessions are all male and recorded with the car stopped;
  they are not sufficient evidence of global generalization.
- Binary-yawn V2 reaches 0.815 frame macro-F1 on held-out subject 9.
- The canonical temporal decoder reaches 0.667 held-out yawn event F1
  (2/4 events, precision 1.0, recall 0.5, no false alarms in the clip).
- Both missed subject-9 events are hand-occluded yawns; both visible-mouth yawns
  are detected. This motivates a small face-context ablation rather than more
  global threshold tuning.

Generated artifacts:

- `artifacts/dmd_partial_manifest.csv`
- `artifacts/dmd_partial_manifest.summary.json`

### Gaze work

- V7 dual-eye affine/FiLM gaze is a candidate auxiliary branch.
- Corrected MPIIGaze V7 LOPO mean: 4.608 degrees 3D angular error.
- MPI-to-Gaze360 zero-shot transfer is poor, so gaze must earn inclusion through
  a DMS ablation. It is not assumed to improve driver-state classification.
- V6 versus V7 selection remains gated by matched preprocessing and metrics.

## 4. Target in-car architecture

```text
20 FPS driver stream
        |
        +-- scene-cut / quality detector
        |
        +-- perception at 4-5 FPS
              |
              +-- MediaPipe landmarks
              +-- affine left/right eye crops -> shared tiny CNN
              +-- affine mouth crop -> tiny CNN
              +-- optional low-resolution face crop -> tiny CNN
              +-- EAR, MAR, pose, crop quality
              +-- frozen winning gaze branch
                              |
                    bounded FiLM fusion
                              |
                     frozen causal GRU
                              |
          auxiliary heads + five-state logits
                              |
              causal owner-calibrated decoder
                              |
           per-frame state expanded to 20 FPS
```

Initial size targets, excluding MediaPipe:

- global model: at most 1.5 million parameters;
- owner adapter: 5,000-30,000 trainable parameters;
- causal history: 2-5 seconds;
- end-to-end CPU latency must include MediaPipe, not just the classifier.

## 5. Execution phases and gates

### Phase 0 - Data and contracts

Status: in progress

Work:

- [x] Audit practice and scored trip schemas.
- [x] Confirm the actual five-class evaluator target.
- [x] Acquire a partial DMD drowsiness subset.
- [x] Build and test raw DMD eye/yawn manifest generation.
- [x] Define one canonical label and event schema.
- [ ] Add competition-trip manifest generation.
- [x] Define how DMD actions map to auxiliary targets.
- [ ] Acquire a small DMD distraction subset when needed.
- [ ] Implement video-cut detection and causal state reset.

Exit gate:

- loaders reproduce frame counts and timestamps;
- no hidden ground-truth field enters model features;
- subject/session identifiers are available for grouped splits;
- raw labels remain traceable to their source annotation.

### Phase 1 - Lightweight global baseline

Status: pending

Experiments:

1. B0: EAR/MAR/head-pose rules and causal duration thresholds.
2. B1: geometry plus a small classifier, no image crops.
3. B2: eye/mouth CNNs, FiLM, and one-layer causal GRU.

Training targets:

- auxiliary eye state;
- auxiliary yawn probability/type;
- auxiliary gaze-away and quality when labels exist;
- five-state competition target on valid labelled clips;
- event boundary and temporal-consistency terms.

Training order:

1. Overfit one small batch.
2. Overfit one subject/session.
3. Run one grouped validation fold.
4. Run all permitted grouped folds only after the first three pass.

Exit gate:

- beats the constant and rule baselines on macro-F1;
- no class has silently collapsed to zero recall;
- predictions are causal and reset correctly at hard cuts;
- latency and model size are recorded end to end.

### Phase 2 - Gaze, crop, FiLM, and temporal ablations

Status: pending

Required comparisons:

- raw crop versus affine-normalized crop;
- feature concatenation versus FiLM;
- no gaze versus V6 gaze versus V7 gaze;
- frame classifier versus causal GRU;
- 4 FPS versus 5 FPS versus 10 FPS perception;
- global model versus owner-neutral calibration.

Primary metrics:

- competition accuracy and macro-F1;
- per-class precision, recall, and F1;
- event onset delay and event-level F1;
- false warnings per hour;
- microsleep and distraction recall;
- full CPU/GPU latency, memory, parameters, and FLOPs.

Exit gate:

- choose the smallest model within one practical margin of the best score;
- gaze is included only if it improves DMS holdout results or robustness;
- the winning configuration passes future-session and lighting checks.

### Phase 3 - Few-shot owner personalization

Status: pending

Adaptation ladder:

| Level | Adaptive state | Default risk |
| --- | --- | --- |
| A0 | owner prototypes, gaze offsets, event thresholds | low |
| A1 | class bias and temperature calibration | low |
| A2 | owner embedding and bounded final FiLM | medium |
| A3 | low-rank adapter after the frozen GRU | high |
| A4 | visual or temporal backbone | prohibited by default |

Always frozen by default:

- MediaPipe and affine crop definitions;
- early and middle visual encoders;
- global gaze estimator;
- primary causal GRU;
- global microsleep safety representation.

Few-shot protocol:

- learn an owner-neutral prototype from 1-3 parked minutes;
- collect safe guided gaze/head directions and natural blinking;
- use naturally observed yawns when available;
- never require an owner to demonstrate microsleep while driving;
- evaluate K-shot curves for K = 1, 5, 10, and 20 events;
- validate on a later owner session and the global safety set.

An adaptation level is accepted only when it improves owner holdout macro-F1
without materially reducing global rare-event recall or increasing false alarms.

### Phase 4 - Cloud update, deployment, and submission

Status: pending

Cloud workflow:

1. Upload only consented, selected uncertain/event clips.
2. Reject corrupt, duplicate, low-quality, and out-of-owner samples.
3. Use global-teacher labels only as candidates, not unquestioned truth.
4. Train only the permitted owner parameter group.
5. Use global replay, distillation, and adapter regularization.
6. Validate against owner holdout, global safety, and the previous version.
7. Sign and send only the adapter/profile delta.
8. Retain automatic rollback in the car.

Competition workflow:

- run causal inference over all ten scored trips;
- expand sampled predictions to every required frame;
- validate allowed class strings, frame order, row count, and timestamps;
- produce submission CSVs and a reproducibility report;
- archive the exact global checkpoint and configuration.

Exit gate:

- all ten CSVs pass the supplied evaluator's schema checks;
- no missing frames or invalid labels;
- deployment package supports owner switching and rollback;
- performance, latency, limitations, and data use are documented.

## 6. Freeze/adapt decision experiment

Run the same owner folds and shots for every row:

```text
G0  frozen global model
A0  G0 + prototypes/thresholds
A1  A0 + bias/temperature
A2  A1 + bounded FiLM
A3  A2 + post-GRU low-rank adapter
```

For each configuration report:

- owner-session macro-F1 and per-class recall;
- global holdout macro-F1 and rare-event recall;
- false alarms per hour;
- expected calibration error;
- parameter count and update payload size;
- improvement confidence interval across owners.

Select the earliest adaptation level whose gain is repeatable. Do not select a
larger adaptive surface merely because it has the best training score.

## 7. Main risks and controls

| Risk | Control |
| --- | --- |
| Few-shot overfitting | tiny adapter, holdout day, bounded FiLM |
| Self-training confirmation | uncertainty filter and verified labels |
| Alert-class domination | balanced replay and per-class reporting |
| Rare-event forgetting | frozen safety head and global replay |
| Identity/session leakage | subject/session grouped research splits |
| DMD-to-competition label mismatch | raw auxiliary targets plus calibrated decoder |
| MediaPipe latency | 4-5 FPS perception, tracking, full latency accounting |
| Hard clip transitions | cut detector and temporal reset |
| Unsafe OTA update | signed versions, validation gate, rollback |
| Privacy exposure | selected clips, retention policy, owner deletion |

## 8. Immediate task queue

1. Define the canonical raw-event and five-state schema.
2. Build the competition-trip manifest beside the DMD manifest.
3. Implement a streaming 4-5 FPS MediaPipe cache without extracting JPGs.
4. Add affine eye and mouth crops with quality masks.
5. Build B0 rules and B1 geometry classifier.
6. Run a one-batch and one-subject overfit test.
7. Add B2 FiLM plus causal GRU.
8. Compare no-gaze, V6, and V7 variants.
9. Implement A0 owner prototypes and threshold calibration.
10. Package the winning global model and first owner adapter separately.

## 9. Definition of the final model

The model is final only when:

- the global configuration wins the registered grouped ablation;
- gaze inclusion is supported by DMS evidence;
- the smallest safe personalization level is selected by owner holdout results;
- rare-event recall does not regress beyond the registered tolerance;
- full in-car latency includes landmark extraction;
- the cloud update can be rejected or rolled back;
- the competition submission and research evaluation are reproducible.
