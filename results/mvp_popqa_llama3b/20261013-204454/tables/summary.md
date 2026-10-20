| Method | Timing | AUROC [95% CI] | AUPRC [95% CI] | ECE | AURC | Risk@80% |
|---|---|---|---|---|---|---|
| probe | pre-generation | 0.897 [0.878, 0.915] | 0.951 [0.939, 0.961] | 0.021 | 0.417 | 0.638 |
| probe_layer0 | pre-generation | 0.500 [0.500, 0.500] | 0.709 [0.685, 0.731] | 0.007 | 0.709 | 0.709 |
| first_token_entropy | pre-generation | 0.878 [0.856, 0.898] | 0.941 [0.926, 0.953] | 0.035 | 0.437 | 0.644 |
| popularity | pre-generation | 0.673 [0.636, 0.708] | 0.769 [0.736, 0.803] | 0.073 | 0.548 | 0.693 |
| metadata | pre-generation | 0.826 [0.800, 0.850] | 0.907 [0.886, 0.927] | 0.045 | 0.469 | 0.649 |
| neg_mean_logprob | post-generation | 0.854 [0.830, 0.877] | 0.921 [0.902, 0.939] | 0.050 | 0.451 | 0.649 |

Test set: n = 1284, error base rate = 0.709. Probe uses layer 15 (selected on validation AUROC).

| Method | Within-relation AUROC (macro) | Relations used |
|---|---|---|
| probe | 0.885 | 7 |
| probe_layer0 | 0.500 | 7 |
| first_token_entropy | 0.852 | 7 |
| popularity | 0.749 | 7 |
| metadata | 0.749 | 7 |
| neg_mean_logprob | 0.802 | 7 |

Metadata alone: AUROC 0.826; metadata + probe: 0.901; ΔAUROC 0.075 [0.055, 0.095].

Ambiguous test examples: 563.

| Method | Headline AUROC | Ambiguous = correct | Ambiguous = incorrect |
|---|---|---|---|
| probe | 0.897 | 0.534 | 0.914 |
| probe_layer0 | 0.500 | 0.500 | 0.500 |
| first_token_entropy | 0.878 | 0.711 | 0.853 |
| popularity | 0.673 | 0.490 | 0.705 |
| metadata | 0.826 | 0.535 | 0.854 |
| neg_mean_logprob | 0.854 | 0.770 | 0.821 |
