# Bonus probe: tampered images (never trained on)

real: 15   tampered: 15

Threshold and calibration are inherited unchanged from the binary real-vs-synthetic model. Nothing here was refitted.


## Flag rate by condition

| condition | flagged tampered (recall) | flagged real (FPR) | AUROC | mean P(fake) tampered |
|---|---|---|---|---|
| clean | 0.6000 | 0.0000 | 1.0000 | 0.210 |
| jpeg_q90 | 0.6667 | 0.0000 | 1.0000 | 0.247 |
| jpeg_q70 | 1.0000 | 0.0000 | 1.0000 | 0.884 |
| jpeg_q50 | 1.0000 | 0.0667 | 1.0000 | 0.933 |
| jpeg_q30 | 1.0000 | 0.3333 | 1.0000 | 1.000 |
| blur_s0.5 | 0.6667 | 0.0000 | 1.0000 | 0.513 |
| blur_s1.0 | 0.9333 | 0.0000 | 0.9911 | 0.835 |
| blur_s2.0 | 1.0000 | 0.0000 | 1.0000 | 0.969 |
| resize_0.5x | 0.8667 | 0.0000 | 1.0000 | 0.866 |
| resize_0.25x | 1.0000 | 0.0000 | 1.0000 | 0.998 |
| noise_s0.02 | 1.0000 | 0.0000 | 1.0000 | 1.000 |
| noise_s0.05 | 1.0000 | 0.0000 | 1.0000 | 1.000 |
| noise_s0.1 | 1.0000 | 0.0000 | 1.0000 | 1.000 |
| color_up | 0.9333 | 0.0000 | 1.0000 | 0.867 |
| color_down | 0.8000 | 0.0000 | 0.9956 | 0.550 |
| color_mixed | 0.7333 | 0.0000 | 1.0000 | 0.623 |
| crop_80 | 1.0000 | 0.0000 | 1.0000 | 0.772 |
| social_repost | 1.0000 | 0.0000 | 1.0000 | 0.939 |
| filtered_share | 1.0000 | 0.0000 | 1.0000 | 0.800 |
| thumbnail_crop | 1.0000 | 0.0667 | 0.9956 | 0.662 |
| lowlight_msg | 1.0000 | 0.0000 | 1.0000 | 0.809 |
| screenshot_chain | 1.0000 | 0.2000 | 1.0000 | 0.945 |

## Detection vs size of the edited region (clean condition)

| edited fraction | n | recall @ deployed threshold | mean score |
|---|---|---|---|
| 5%–10% | 10 | 0.8000 | -0.156 |

Pearson r between edited fraction and detector score: **0.746**
A clearly positive r is the headline: the detector only sees a manipulation once it is large enough to survive the 224px resize.