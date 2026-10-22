# SNAPSHOT — task_b เดิมจาก CN340_Stock_Prediction

> **ห้ามแก้ไฟล์ในโฟลเดอร์นี้** (`task_b/legacy/`) · ใช้แบบอ่านอย่างเดียว · ถ้าต้องเปลี่ยนพฤติกรรมให้เขียน wrapper ใน `task_b/phase2/`

| รายการ | ค่า |
|---|---|
| ที่มา | https://github.com/panifield/CN340_Stock_Prediction (โฟลเดอร์ `task_b/`) |
| commit | `e49bbdd90a47609cbc1a2125da54a7f516123e3a` (Merge pull request #3 from panifield/task_b/fix-daily-snapshot-eol) |
| วันที่ copy | 2026-10-18 |
| วิธี copy | `git cat-file blob HEAD:<path>` ลงไฟล์ตรง ๆ (ไม่ผ่านการแปลง line ending) แล้วตรวจว่า `git hash-object --no-filters` ของไฟล์ที่ copy = blob id ใน commit ทุกไฟล์ (41 ไฟล์ ไม่รวม SNAPSHOT.md นี้) |

## ข้อควรระวังเรื่อง line ending

- ไฟล์โค้ด/แผน/ผลลัพธ์ใน commit นี้เป็น **LF**; ไฟล์ข้อมูล `raw_data/*.csv` และ `raw_data_intraday/*.csv` เป็น **CRLF** (ตามต้นฉบับ) ส่วน `raw_data_daily/*.csv` เป็น LF
- SHA-256 ด้านล่างคิดจากไบต์ของ blob ตรง ๆ · เครื่องที่ตั้ง `core.autocrlf=true` ทำให้ checkout ไฟล์ที่ไม่มี attribute `-text` เป็น CRLF → SHA เปลี่ยน
- `legacy/.gitattributes` (copy มาจากต้นฉบับ) กันเฉพาะไฟล์ข้อมูล · เพื่อกันไฟล์โค้ด/แผน/ผลด้วย จึงเพิ่ม `legacy/** -text` ใน `task_b/.gitattributes` (อยู่นอก `legacy/` จึงไม่แก้ไฟล์ที่ snapshot)

## ไฟล์ที่ copy และ SHA-256

| ไฟล์ (ใต้ `task_b/legacy/`) | ไบต์ | SHA-256 |
|---|---|---|
| `.gitattributes` | 916 | `2cf567effc4e0bdd5b6e0a8f264a9bbfd5deac0fade2667b5b1467cc5a035382` |
| `baselines.py` | 1943 | `56fb2c376bb0088cad728e6ba5ebf61fdbd1e73c2eddd063654683080f77b29a` |
| `config.py` | 17420 | `e6bb7ee9c9978912490f757be1872e2a4568f84c1a5aaad00006214e63fb076d` |
| `daily_close.py` | 13626 | `f836bc133ebf876c463236501a636a2c52aea92101b9ef603e9cd616f9a30e96` |
| `data_loader.py` | 5147 | `1bfb81b9166d254f66ece2d5b62686c9993c00a74bc27ae842c6c4c712285269` |
| `diagnostics.py` | 4157 | `dc9060300fdbeca25ab2d6630e3173d6f413ae840e74e5d981d10973801c072c` |
| `evaluate.py` | 8497 | `d91e0a2d065e2258e6b163adfadc4e89314fde8110711ebcff6cbde6141b098d` |
| `features.py` | 13306 | `dda9cf8b1c17c87160f8a811d65d897a734958239e94bb174c0cd08750354dc0` |
| `intraday_1600.py` | 16679 | `7a4e44e8ea2d489f86f231856119519d5a3b8a249eaf4f4f51ca0a1a862550d4` |
| `metrics_1600.py` | 5584 | `2c7fd01c5bfbf1f2388479c178aee2929c5bfdbe160a4d2f38f70038faa28b3e` |
| `models.py` | 8217 | `54eec4dbea27c3ca2831676d6c4013d0de712565f8dc98cca8552a187543017b` |
| `PHASE1D_PLAN.md` | 21755 | `04f069901f461e0e6ec027af21979dcbbe75fa9776eb2cd39bb18606448ec04d` |
| `PHASE1D_PLAN_R2.md` | 28916 | `bada95c8300411b23a521c7b8e108ddd9166a7bd4ed1c67a9e445b5fe47b4750` |
| `raw_data/ADVANC_10Y_Cleaned.csv` | 122181 | `3ab9dd961be3265bc7cd4c1d488bdb4ef7ce0de7262c59d2a126bd2899378a31` |
| `raw_data/KBANK_10Y_Cleaned.csv` | 122899 | `431e2ee6b19444721bc8b74e03b13fccf1361149e1538a8ecc806279605019b3` |
| `raw_data_daily/ADVANC_BK_1d_locked.csv` | 61645 | `641dab045cb5f5141bbdefdf41adc066ba4686a6d27d76ab895443a51a0ca800` |
| `raw_data_daily/KBANK_BK_1d_locked.csv` | 62579 | `ab244d8ff4f3b3c6dff3329aa4d5e8ae94981eeb57bc5cd64473deebe664ec31` |
| `raw_data_daily/README.md` | 2324 | `933e0b45a635e79eb885060c8710e52b5fd46fc34002f4a5e0e5e36e7d614ea8` |
| `raw_data_intraday/ADVANC_BK_1h_730d.csv` | 312741 | `2097bb53ce1c6ad49023f432128b39363187f4fdaebcb12bbd5ee7e5a295e79c` |
| `raw_data_intraday/KBANK_BK_1h_730d.csv` | 314225 | `b192ad2c0bc3ae2a75b732752369c56b7dd27d043900eb6e8fb2d5fb08912060` |
| `raw_data_intraday/README.md` | 2095 | `0e350f7bed543279ed79617d8344a7b6740b355c4512fefa6d9d25484076d819` |
| `requirements.txt` | 308 | `9accefefbdebb77ab6c5700a52bf5c935e4c08f33f6651ced917f20ee967b980` |
| `results/phase1d_r2/backfill_leak_test.csv` | 7648 | `8ef2b191fd6bb3ceb432c2001697ab777d4b81f4f41fb1bc0519ea05a64570b9` |
| `results/phase1d_r2/backfill_same_day_close.csv` | 62913 | `5158a7e901c09eaace009a5ce63ad183f4803c4655e14aa2eefebda736a0de62` |
| `results/phase1d_r2/phase1d_baselines.csv` | 3761 | `be5cdbb7bf5482bc54d206ec84dbda5f9f311d5cfdc3e038b2cbe9d5435107bb` |
| `results/phase1d_r2/phase1d_econ.csv` | 2187 | `bcd2b95e306b4e00f12422d12894573fb96c67c22a8de46847e6c0f91d2195b5` |
| `results/phase1d_r2/phase1d_evals.csv` | 51117 | `02afb2ba93dfbc33602a9446f52b4d90c4018735f5765e2958d3822442020678` |
| `results/phase1d_r2/phase1d_folds.csv` | 456 | `5795d15caeffa7b8a9536e0579adb44733072afca4159e4d69db41cc46d371fe` |
| `results/phase1d_r2/phase1d_scores.csv` | 3618 | `3c23e403b634e4d1551cc96df6d6d7257cfda4a2c000c829ed1eb4b293697ed1` |
| `results/phase1d_r2/phase1d_summary.md` | 1356 | `c2a1a89c54d8b80736f18a3144cb2d9151c4450e690818e84b15442ea168e261` |
| `results/tuning_scores.csv` | 3803 | `75ddf7a6bb6cfe9bfc05e9ae97dec4af79510bc28bc32c8803aae33215943368` |
| `results/tuning_stage1_halflife.csv` | 9453 | `1f46d70d8be845f3d56aa8684d06eae94d64d418873b4bdb1af077aa8ae89142` |
| `results/tuning_stage2_params.csv` | 26754 | `d935d58ba4f16521a413f064503ef8c9922b07a9641e6e2b2ed0af1c3de115d6` |
| `results/tuning_summary.md` | 15967 | `e408289194f4bfe9475cf48271744d0dc69b74b90d24221879a9cbb329c345b6` |
| `splits.py` | 5646 | `41a6c0e57d05e289db38db38fbf064342b0b89a5d5a15078168482247971aba7` |
| `targets.py` | 2626 | `e4eb0f00ce0815f3dd939ead8287838ee1e73bdea47cb27383ef000ff174b8c3` |
| `trading_costs.py` | 8466 | `fbcf1dae9003d21395a9a1ee815c936f3bac3246fad601b7267c757d6d402176` |
| `tune.py` | 23875 | `6024752707d0c55c23e34a31dacefca6a8522528287b9ed8462cc0d67b4fd3d1` |
| `tune_1600.py` | 19419 | `7990ccda88f91db2998688eb9eea5c10a4305accebc1e609e55c172b9680482f` |
| `TUNING_PLAN.md` | 11150 | `9f69dd991699ad70661c315a01710b2279af7c779c0d5a3a60b66bf163f2ed85` |
| `weighting.py` | 3590 | `2c25308dc5209a774411a5e7bfeb28f8e7ecc81b1e6adf7535a4358ce14d4a5e` |

## ไฟล์ที่ไม่ได้ copy (ตั้งใจ)

- ส่วน live/automation: `predict_*.py`, `daily_1600.py`, `daily_next_day.py`, `live_1600.py`, `record_outcomes.py`, `report_*.py`, `results/prediction_log*.csv`, `results/outcomes.csv`, `automation/`, `tools/`, `raw_data_*_live/`, `raw_data_sources/`
- `set_holidays.txt` — `daily_close.py` อ่านไฟล์นี้เฉพาะในเส้นทาง live (`require_previous_close` / `read_holidays`) ซึ่ง dev replay ไม่เรียก (`check_previous=False`) · ไม่ได้ import และไม่ได้อ่านตอน import โมดูล · ถ้าจะใช้ต้องถามก่อน
- `main.py`, `curves.py`, `tests/` ของ legacy, `results_reference/`, `tools/make_locked_daily.py` (อยู่ใน `LOCKED_FILES_R2` ของ `tune_1600.py` แต่เฉพาะเส้นทาง `--run` / `check_preregistered` ซึ่งห้ามเรียก)

## ผลตรวจ import closure

โมดูลทั้ง 16 ตัว (`config`, `data_loader`, `features`, `targets`, `splits`, `models`, `weighting`, `evaluate`, `baselines`, `trading_costs`, `diagnostics`, `tune`, `intraday_1600`, `metrics_1600`, `daily_close`, `tune_1600`) import ได้เมื่อมีเฉพาะ `task_b/legacy` ใน `sys.path` · ไม่มีโมดูลนอกรายการถูก import (ตรวจแล้ว 2026-10-18)

## ผลตรวจข้อมูลราคารายวัน เทียบกับ `raw_data/` ที่ root ของ StockCNN

| หุ้น | SHA-256 ของ `raw_data/` ที่ root | ผล |
|---|---|---|
| KBANK | `c2068d4ce1c71db799a361b9103fe563bffc8adde418629fac01e5106fbe61f6` | **ไบต์ 121,298 ตัวแรกของไฟล์ใน legacy เหมือนไฟล์ที่ root ทุก byte** (ไฟล์ root จบที่ 2026-08-28) |
| ADVANC | `cbd5db1d6dbc3dd0bce59266825ee11e8cb49e12d2bcb425a7d33e4300933c68` | **ไบต์ 120,595 ตัวแรกเหมือนทุก byte** (ไฟล์ root จบที่ 2026-08-28) |

- ไฟล์ใน legacy ถูก append ต่อถึง 2026-10-14 (32 แถว: 2026-08-31 … 2026-10-14) ซึ่งไม่ตรงกับ root ตามที่คาด
- โหมด next_day ตัดที่ `val_end` = 2025-02-25 จึงไม่แตะแถวเหล่านี้ · โหมด 16:00 ใช้ `raw_data_daily/*_1d_locked.csv` (ล็อก SHA, ถึง 2026-09-25) ซึ่งสร้างจากไฟล์ legacy ที่ append แล้ว — **ห้ามใช้ `raw_data/` ที่ root แทนในโหมด 16:00** เพราะไม่มีช่วง 2026-08-31 … 2026-09-25
