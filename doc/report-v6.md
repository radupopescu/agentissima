| Configuration | Driver | Suite W | Suite T | TTFT | Gen tok/s | Prompt tok/s | Peak RAM | Swap | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| BON-G2 | native | - | - | 27.710 | 27.1 | 232.8 | 7.53 GiB | yes | passed Stage 0 only |
| BON-G2 | pi | 0/30 | - | 27.710 | 27.1 | 232.8 | 7.53 GiB | yes | excluded: failed Stage 2A gate |
| BON-M2 | native | - | - | 27.442 | 34.6 | 235.3 | 10.02 GiB | yes | passed Stage 0 only |
| BON-M2 | pi | 3/30 | - | 27.442 | 34.6 | 235.3 | 10.02 GiB | yes | excluded: failed Stage 2A gate |
| LFM-BF16 | native | - | - | 8.978 | 28.9 | 758.6 | 7.65 GiB | no | passed Stage 0 only |
| LFM-BF16 | pi | 19/30 | 26/30 | 8.978 | 28.9 | 758.6 | 7.65 GiB | no | proceeded to Stage 2B |
| LFM-G8 | native | 9/30 | 20/30 | 8.616 | 53.1 | 770.3 | 3.52 GiB | no | proceeded to Stage 2B |
| LFM-G8 | pi | 22/30 | 27/30 | 8.616 | 53.1 | 770.3 | 3.52 GiB | no | proceeded to Stage 2B |
| LFM-GQ4 | native | 12/30 | 27/30 | 8.581 | 83.3 | 769.0 | 2.32 GiB | no | proceeded to Stage 2B |
| LFM-GQ4 | pi | 19/30 | 26/30 | 8.581 | 83.3 | 769.0 | 2.32 GiB | no | proceeded to Stage 2B |
| LFM-M8 | native | - | - | 14.601 | 49.9 | 454.7 | 4.78 GiB | no | passed Stage 0 only |
| LFM-M8 | pi | 17/30 | 25/30 | 14.601 | 49.9 | 454.7 | 4.78 GiB | no | proceeded to Stage 2B |
