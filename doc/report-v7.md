| Configuration | Driver | Suite W | Suite T | TTFT | Gen tok/s | Prompt tok/s | Peak RAM | Swap | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| BON-G2 | native | - | - | 27.569 | 27.2 | 234.0 | 7.60 GiB | no | passed Stage 0 only |
| BON-G2 | pi | 0/30 | - | 27.569 | 27.2 | 234.0 | 7.60 GiB | no | excluded: failed Stage 2A gate |
| BON-M2 | native | - | - | 27.427 | 34.6 | 235.4 | 10.02 GiB | yes | passed Stage 0 only |
| BON-M2 | pi | 3/30 | - | 27.427 | 34.6 | 235.4 | 10.02 GiB | yes | excluded: failed Stage 2A gate |
| LFM-BF16 | native | - | - | 8.991 | 28.8 | 758.3 | 7.65 GiB | no | passed Stage 0 only |
| LFM-BF16 | pi | 20/30 | 26/30 | 8.991 | 28.8 | 758.3 | 7.65 GiB | no | proceeded to Stage 2B |
| LFM-G8 | native | 9/30 | 18/30 | 8.622 | 52.9 | 769.8 | 3.52 GiB | no | proceeded to Stage 2B |
| LFM-G8 | pi | 20/30 | 29/30 | 8.622 | 52.9 | 769.8 | 3.52 GiB | no | proceeded to Stage 2B |
| LFM-GQ4 | native | 12/30 | 25/30 | 8.587 | 83.0 | 769.1 | 2.33 GiB | no | proceeded to Stage 2B |
| LFM-GQ4 | pi | 20/30 | 26/30 | 8.587 | 83.0 | 769.1 | 2.33 GiB | no | proceeded to Stage 2B |
| LFM-M8 | native | - | - | 14.550 | 50.2 | 456.2 | 5.86 GiB | no | passed Stage 0 only |
| LFM-M8 | pi | 20/30 | 28/30 | 14.550 | 50.2 | 456.2 | 5.86 GiB | no | proceeded to Stage 2B |
