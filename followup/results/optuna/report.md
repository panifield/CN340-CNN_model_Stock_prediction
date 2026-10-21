# Optuna hyperparameter search (Task A)

- ค้นหา 50 trial (TPE) บน walk-forward selection เท่านั้น (80 วัน/หุ้น x 2 หุ้น = 160 วัน)
- best trial #42: selection acc 0.575 (2 seed ระหว่างค้นหา)
- เกณฑ์ Bonferroni = 0.05 / 50 = 0.00100 -> p ของ best (3 seed) = 0.0894 -> **ไม่ผ่าน**

## ค่าที่ดีที่สุด

```json
{
  "transform": "stft_complex",
  "nperseg": 6,
  "ch_ret_close": false,
  "ch_gap_open": false,
  "ch_range_hl": true,
  "ch_chg_vol": false,
  "ch_parity_sig": true,
  "ch_int_chg": false,
  "extras": true,
  "window": 30,
  "rounds": true,
  "arch": "current",
  "noise": 0.1,
  "lr": 0.0009475997100047405,
  "weight_decay": 0.001784281408984759,
  "batch_size": 8,
  "bottleneck": 0,
  "skip": "mlp"
}
```

## ผลเทียบ

| โมเดล | selection (ใช้เลือก) | p | holdout | p | replication 9 ปี | n | p |
|---|---|---|---|---|---|---|---|
| optuna best | 0.556 | 0.0894 | **0.469** | 0.734 | **0.500** | 2016 | 0.527 |
| base | 0.525 | 0.29 | **0.516** | 0.45 | **0.504** | 2016 | 0.352 |
| hybrid_extras | 0.575 | 0.0343 | **0.609** | 0.0517 | **0.505** | 2016 | 0.32 |

## ความสำคัญของ hyperparameter (fANOVA)

| param | importance |
|---|---|
| batch_size | 0.196 |
| weight_decay | 0.191 |
| window | 0.146 |
| ch_gap_open | 0.113 |
| ch_range_hl | 0.101 |
| nperseg | 0.064 |
| bottleneck | 0.057 |
| lr | 0.037 |
| noise | 0.033 |
| arch | 0.016 |
