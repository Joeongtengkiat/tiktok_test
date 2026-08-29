# Bonus probe: tampered images (never trained on)

real: 15   tampered: 15

Threshold and calibration are inherited unchanged from the binary real-vs-synthetic model. Nothing here was refitted.


## Flag rate by condition

| condition | flagged tampered (recall) | flagged real (FPR) | AUROC | mean P(fake) tampered |
|---|---|---|---|---|
| clean | 0.0000 | 0.0000 | 0.9956 | 0.000 |
| jpeg_q90 | 0.0000 | 0.0000 | 1.0000 | 0.000 |
| jpeg_q70 | 0.6000 | 0.0000 | 0.9733 | 0.290 |
| jpeg_q50 | 0.7333 | 0.0000 | 0.9644 | 0.347 |
| jpeg_q30 | 0.8000 | 0.0000 | 0.9644 | 0.663 |
| blur_s0.5 | 0.0667 | 0.0000 | 0.9822 | 0.000 |
| blur_s1.0 | 0.4000 | 0.0000 | 0.9511 | 0.099 |
| blur_s2.0 | 0.7333 | 0.0000 | 0.9956 | 0.349 |
| resize_0.5x | 0.6000 | 0.0000 | 0.9644 | 0.165 |
| resize_0.25x | 0.8667 | 0.0000 | 1.0000 | 0.236 |
| noise_s0.02 | 0.7333 | 0.0000 | 1.0000 | 0.253 |
| noise_s0.05 | 0.8667 | 0.0000 | 1.0000 | 0.406 |
| noise_s0.1 | 0.8667 | 0.0000 | 0.9867 | 0.366 |
| color_up | 0.2667 | 0.0000 | 0.9822 | 0.064 |
| color_down | 0.2667 | 0.0000 | 0.9956 | 0.061 |
| color_mixed | 0.0667 | 0.0000 | 0.9244 | 0.001 |
| crop_80 | 0.8667 | 0.0000 | 1.0000 | 0.520 |
| social_repost | 0.6667 | 0.0000 | 1.0000 | 0.328 |
| filtered_share | 0.4667 | 0.0000 | 0.9778 | 0.067 |
| thumbnail_crop | 0.8667 | 0.2000 | 0.9200 | 0.396 |
| lowlight_msg | 0.5333 | 0.0000 | 0.9911 | 0.192 |
| screenshot_chain | 0.9333 | 0.0000 | 0.9956 | 0.553 |

## Detection vs size of the edited region (clean condition)

| edited fraction | n | recall @ deployed threshold | mean score |
|---|---|---|---|
| 5%–10% | 10 | 0.0000 | -0.187 |

Pearson r between edited fraction and detector score: **0.556**
A clearly positive r is the headline: the detector only sees a manipulation once it is large enough to survive the 224px resize.