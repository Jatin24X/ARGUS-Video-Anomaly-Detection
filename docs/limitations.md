# Limitations

- Full-frame features can miss small or spatially localized anomalous objects.
- A model trained for one camera domain may not calibrate well on another.
- VideoMAE inference is computationally heavier than lightweight CNN alternatives.
- Scale-to-zero deployment adds cold-start latency.
- AUC measures ranking quality but does not define an operational alert threshold.
- Avenue and UBnormal differ in scene construction and anomaly distribution, so cross-dataset numbers should not be merged into one leaderboard.

## Next Technical Step

The highest-value extension is a controlled localization branch that preserves the frame-centric baseline and tests whether region-aware evidence improves small-object anomalies. It should be evaluated as an ablation, not silently folded into the existing claim.
