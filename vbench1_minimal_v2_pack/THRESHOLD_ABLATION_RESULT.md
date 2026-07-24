# Dynamic Switch Threshold Calibration Result

The calibration used 20 VBench-1.0 prompts, one fixed seed per prompt, and thresholds
`0.10`, `0.15`, `0.20`, `0.25`, and `0.30`. All routing, model, resolution, frame, and
sampling settings were held fixed.

| Threshold | Seven-dimension VBench mean | Relative quality drop | Mean switch step |
| ---: | ---: | ---: | ---: |
| 0.10 | 0.8837 | 0.000% | 28.25 |
| 0.15 | 0.8835 | 0.026% | 28.00 |
| 0.20 | 0.8835 | 0.026% | 28.00 |
| 0.25 | 0.8835 | 0.026% | 28.00 |
| 0.30 | 0.8835 | 0.026% | 28.00 |

Thresholds `0.15` through `0.30` produced byte-identical MP4 files for all 20 prompts
and therefore represent the same quality/compute plateau on this calibration set.
Threshold `0.10` switched later on three prompts (two at step 30 and one at step 29),
for an average of 0.25 additional full-large-model steps, while improving the aggregate
score by only 0.026%.

The selected default is **0.15**. It reaches the earliest observed switching schedule,
while using the smallest threshold on the tied plateau. Larger thresholds provide no
measured compute benefit here and may switch unnecessarily early on unseen prompts.

Raw wall-clock generation times are not used for the final selection because the run
was externally interrupted and resumed under substantially different server loads.
