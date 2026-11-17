# SET15 untrained table v2 (numbers from report61.py; 15 frozen games)

Untrained Qwen 27B, thinking on, T 0, no paragraph. Normal budget 35, long budget 65. B9 = complete-evidence share (exact top >= .95 at any turn). Premature = answered with exact top < .95 while moves were left. "r|prem" = right among premature answers.

| k | game | version | board | plays | errors | right | moves | budget hits | B9 share | premature | r\|prem | cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | face | v6.0 face (e3aa28da) | normal | 40 | 0 | 0.450 | 30.8 | 0 | 0.300 | 0.500 | 0.25 | root N cell |
| 1 | face | v6.0 face (e3aa28da) | long | 20 | 0 | 0.150 | 58.9 | 0 | 0.000 | 0.800 | 0.0625 | root L l-face: the root L face file 8d63f53d (a long-board fix; its normal worlds, rules text and budget = root N) is NOT the frozen file e3aa28da, so same_file is False by design; long seeds 50,200,000-019 NOT paired with the normal seeds (NOT the frozen file: see cell note) |
| 2 | jigsaw | v6.0 jigsaw (d7afad4f) | normal | 40 | 0 | 0.325 | 19 | 0 | 0.075 | 0.900 | 0.2778 | root N cell |
| 2 | jigsaw | v6.0 jigsaw (d7afad4f) | long | 20 | 0 | 0.350 | 57.5 | 0 | 0.000 | 0.850 | 0.3529 | root N long cell; OUTSIDE the 1-20 % long band: label only, no drop (training is normal-only) |
| 3 | kaleido | v6.0 kaleido (1abf38bb) | normal | 40 | 0 | 0.475 | 27.9 | 0 | 0.375 | 0.600 | 0.2917 | root N cell |
| 3 | kaleido | v6.0 kaleido (1abf38bb) | long | 20 | 0 | 0.050 | 49.7 | 0 | 0.050 | 0.950 | 0.0526 | root N long cell |
| 4 | bird | bird v1.4 (f77d4876) | normal | 40 | 0 | 0.425 | 32.6 | 0 | 0.225 | 0.300 | 0.4167 | root N2 cell |
| 4 | bird | bird v1.4 (f77d4876) | long | 20 | 0 | 0.200 | 57.2 | 0 | 0.000 | 0.800 | 0.25 | root D1 ld1-bird |
| 5 | fish | fish v1.1 (0b7ea821) | normal | 60 | 0 | 0.650 | 29.5 | 0 | 0.500 | 0.150 | 0.3333 | root N2 cell 40 + 3rd run 20 = POOLED 60 (B9 edge rule) |
| 5 | fish | fish v1.1 (0b7ea821) | long | 20 | 0 | 0.200 | 57.6 | 0 | 0.000 | 0.950 | 0.1579 | root D1 ld1-fish |
| 6 | recipe | recipe v7.1 (6bcd2257) | normal | 40 | 0 | 0.425 | 25 | 1 | 0.375 | 0.525 | 0.0952 | root R71 cell |
| 6 | recipe | recipe v7.1 (6bcd2257) | long | 20 | 0 | 0.050 | 50.5 | 0 | 0.000 | 0.900 | 0.0556 | root D2 ld2-recipe |
| 7 | turtle | turtle (batch S1) (f8d3dde8) | normal | 40 | 0 | 0.625 | 30.1 | 0 | 0.300 | 0.450 | 0.3333 | root S1 cell |
| 7 | turtle | turtle (batch S1) (f8d3dde8) | long | 20 | 0 | 0.250 | 59.8 | 0 | 0.050 | 0.800 | 0.25 | root D2 ld2-turtle; OUTSIDE the 1-20 % long band: label only, no drop (training is normal-only) |
| 8 | balloon | balloon v1.2 (ce8bb559) | normal | 40 | 0 | 0.350 | 30.4 | 0 | 0.075 | 0.700 | 0.25 | root F3 cell |
| 8 | balloon | balloon v1.2 (ce8bb559) | long | - | - | - | - | - | - | - | - | PENDING (not run yet): LONG-FIX ld-balloon (50,171,400) |
| 9 | rocket | rocket v2 (38ea3dd8) | normal | 40 | 0 | 0.400 | 31.1 | 0 | 0.025 | 0.725 | 0.4483 | root F2 cell |
| 9 | rocket | rocket v2 (38ea3dd8) | long | - | - | - | - | - | - | - | - | PENDING (not run yet): LONG-FIX ld-rocket (50,181,400) |
| 10 | bell | bell v2 (8d74f7a6) | normal | 40 | 0 | 0.325 | 30.6 | 0 | 0.150 | 0.575 | 0.2609 | root F4 cell |
| 10 | bell | bell v2 (8d74f7a6) | long | - | - | - | - | - | - | - | - | not available: the long board is not ready (min looks 2) |
| 11 | bee | bee v2 (16030907) | normal | 40 | 0 | 0.350 | 32.6 | 0 | 0.050 | 0.475 | 0.4211 | root F4 cell |
| 11 | bee | bee v2 (16030907) | long | - | - | - | - | - | - | - | - | PENDING (not run yet): LONG-FIX ld-bee (50,201,400) |
| 12 | candle | candle v2 (207a812c) | normal | 40 | 0 | 0.450 | 32.8 | 0 | 0.125 | 0.350 | 0.3571 | root F4 cell |
| 12 | candle | candle v2 (207a812c) | long | - | - | - | - | - | - | - | - | not available: the long board is not ready (min looks 2) |
| 13 | sailboat | sailboat (batch4-rb7) (9d6a6ab0) | normal | 40 | 0 | 0.675 | 30.9 | 0 | 0.350 | 0.300 | 0.4167 | root F6 cell |
| 13 | sailboat | sailboat (batch4-rb7) (9d6a6ab0) | long | - | - | - | - | - | - | - | - | not available: the long board is not ready (B7 min looks 2) |
| 14 | barn | barn v1.3 (be7ddbec) | normal | 60 | 0 | 0.650 | 31.7 | 0 | 0.467 | 0.217 | 0.2308 | root F5c cell 40 + 3rd run 20 = POOLED 60 (B9 edge rule) |
| 14 | barn | barn v1.3 (be7ddbec) | long | - | - | - | - | - | - | - | - | not available: min informative looks 2 in 8 / 1000 fresh long games (b7many check, 72,100,000 x 1000), the same reason as bell / candle |
| 15 | lantern | lantern v1.2 (batch2-rb8-v1.2) (d7db443e) | normal | 40 | 0 | 0.475 | 27.9 | 0 | 0.250 | 0.600 | 0.2917 | root F7 cell |
| 15 | lantern | lantern v1.2 (batch2-rb8-v1.2) (d7db443e) | long | - | - | - | - | - | - | - | - | PENDING (not run yet): LONG-FIX ld-lantern (50,241,400; long READY) |
