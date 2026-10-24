# SOURCES — ที่มาของข้อมูลใน task_b_v2/data/

> บันทึกเมื่อ 2026-10-22 (C0) · **refresh 2026-10-23 (ก่อน commit 2 · DECISIONS B35/B36)** · ไฟล์ CSV ใน data/ ห้ามแก้ · เก็บแบบ byte-for-byte (`data/*.csv -text` ใน `.gitattributes`)

## ราคารายวัน (ใช้ทั้ง next_day และเป็น input ของ 16:00)

| ไฟล์ | แถว | ช่วง | แถวในโซน dev (≤ 2025-02-25) | SHA-256 (เต็ม) |
|---|---|---|---|---|
| `KBANK_10Y_Cleaned.csv` | 2,470 | 2016-08-26 → 2026-10-22 | 2,071 | `97fa5be2b1bfc2afd9da49f74f70360adb2412ca84d90c397874f4540a278082` |
| `ADVANC_10Y_Cleaned.csv` | 2,470 | 2016-08-26 → 2026-10-22 | 2,071 | `4a245577ea0178c5edecf28c3e747f06059841f4ebd49e2a830213ebc8387c1d` |

- คอลัมน์: `Date,Price,Open,High,Low,Vol. ('000),Change %` · วันที่รูปแบบ `%m/%d/%Y` · `Price` = ราคาปิด
- ตรวจแล้ว: เรียงเวลา · ไม่มีวันซ้ำ · ไม่มีค่าว่าง · สองหุ้นมีปฏิทินวันซื้อขายชุดเดียวกัน (ทั้งสองไฟล์)

### ที่มา (ฉบับที่ใช้อยู่ · refresh 2026-10-23)
- repo ทีม `panifield/CN340_Stock_Prediction` · path `task_b/raw_data/<TICKER>_10Y_Cleaned.csv`
- copy จาก clone ในเครื่องผู้ใช้ (`CN340/StockPrediction2`, HEAD = `c87a7f26a2c346f128714f41dbc402172042b218`
  วันที่ 2026-10-23 04:32 +07:00, "chore: A/A2/B outcomes through 2026-10-22")
  ด้วย `git cat-file blob HEAD:task_b/raw_data/<file>` (ไม่ผ่านการแปลง line ending)
- commit ล่าสุดที่แก้ไฟล์ข้อมูล: `6350406b972048aae14312b419a034348a85c9a2`
  (2026-10-23 04:31 +07:00, "data: task_b raw_data through 2026-10-22 (yahoo)")
- ยืนยัน byte-for-byte: `git hash-object --no-filters` ของไฟล์ที่ copy = blob id ใน commit
  (`KBANK` = `a924be69b7e01e7c932cbb396a715ff9f25c3374` · `ADVANC` = `c52775d26ace3a721ca130fcb87f700dc6b7f1f9`) · line ending ยังเป็น CRLF ตามต้นฉบับ

### บันทึกการ refresh (2026-10-23 · ก่อน commit 2)

ฉบับก่อน refresh: 2,467 แถว · 2016-08-26 → **2026-10-19** · copy จาก HEAD `2707d6a` (2026-10-20 18:13 +07:00)
commit ที่แก้ไฟล์ `f52b9d9ad7f8b240da336dd4de25179cbb8044d5` · SHA-256
`dce93acf5a9544bda63b6ee80ed709a287bc83f7a088723e6ad5823899baca1a` (KBANK) ·
`6d3398ba6537bdf5702cda5558157ab41c5b582c8af959493656f6f308eb6625` (ADVANC)

การตรวจตอน refresh (DECISIONS B1 — กันข้อมูลย้อนหลังถูกแก้):

| ตรวจ | ผล |
|---|---|
| 2,467 แถวแรก (header + 2,467 แถวข้อมูล) ตรงกับฉบับก่อน refresh ทุกไบต์ | **ตรงกัน** ทั้งสองหุ้น |
| แถวที่เพิ่ม | **3 แถว** ต่อหุ้น: 10/20/2026, 10/21/2026, 10/22/2026 |
| แถวรวมหลัง refresh | 2,470 ต่อหุ้น |
| แถวในโซน dev (≤ 2025-02-25) | **2,071 เท่าเดิม** · วันสุดท้าย = 2025-02-25 พอดี |
| แถวในโซน final (2025-02-26 → 2026-08-28) | **362 เท่าเดิม** |
| แถวในโซน replay (2026-09-28 → 2026-10-22) | **17 วันทำการ** (B36) |

ผลกระทบต่อ dev และผลทุก Stage: **ไม่มี** — โซน dev ไม่เปลี่ยนแม้แต่แถวเดียว และทุกอย่างก่อน LOCK ตัดข้อมูลหลัง 2025-02-25 ทิ้งก่อนสร้าง feature/ภาพ

### ตรวจความสอดคล้องข้ามแหล่ง
| เทียบกับ | แถวที่เทียบ | ผล |
|---|---|---|
| ฉบับก่อน refresh (จบ 2026-10-19) | 2,467 แถวแรก | ตรงกันทุกแถว (ทั้งสองหุ้น) |
| `task_b/legacy/raw_data/*.csv` ของ branch `task_b` (snapshot, จบ 2026-10-14) | 2,465 แถวแรก | ตรงกันทุกแถว (ทั้งสองหุ้น) · ตรวจ 2026-10-22 |
| `StockCNN/raw_data/*.csv` ที่ root (จบ 2026-08-28) | 2,433 แถวแรก | ตรงกันทุกแถว (ทั้งสองหุ้น) · ตรวจ 2026-10-22 |

## ราคารายชั่วโมง (เฉพาะโหมด 16:00 · `data/intraday/` · copy 2026-10-24 · DECISIONS B41)

- **ที่มา:** repo ทีม `CN340_Stock_Prediction` (clone ในเครื่องผู้ใช้ `CN340/StockPrediction2`, HEAD = `c87a7f26a2c346f128714f41dbc402172042b218`
  2026-10-23 04:32 +07:00) · path เดิม `task_b/raw_data_intraday/` และ `task_b/raw_data_intraday_live/<วันที่>/` → คงโครงโฟลเดอร์เดิมใต้ `data/intraday/`
  · ไฟล์ล็อก 730d ถูก commit ในทีมครั้งล่าสุดที่ `9a1d92584d2671e467cdb93004a60cd8ea753602` ("data: preserve intraday files byte-for-byte")
- **วิธี copy:** `git cat-file blob HEAD:task_b/<path>` (ไม่ผ่านการแปลง line ending) · ยืนยันว่าไบต์ของไฟล์ที่ copy = blob ของทีม = ไฟล์ใน working tree ของทีม ทุกไฟล์
  · `.gitattributes` มี `data/intraday/** -text` → git เก็บแบบ byte-for-byte
- **เก็บเฉพาะไฟล์ที่ `mode1600/run1600` ใช้จริง** (46 ไฟล์): ไฟล์ล็อก 730d 2 หุ้น · snapshot ทาง (i) `20261023/*_035532` · snapshot ทาง (ii) 10 วัน (snapshot ล่าสุดที่ดาวน์โหลด 16:00–16:30 ของวันนั้น ตามกฎ M13) · พร้อม `.meta.json` ทุกไฟล์
  · snapshot อื่นของทีม (เช่น 09-30 16:19 / 16:42, 09-28 23:54, 10-06 18:27) **ไม่ได้ copy** — ไม่ถูกใช้ (ตรวจแล้วว่ากฎ (ii) บนสำเนานี้เลือกไฟล์เดียวกับบน repo ทีมทั้ง 17 วันของทั้งสองหุ้น)
  · `mode1600/AUDIT.md` สร้างจาก repo ทีมฉบับเต็ม (30 snapshot + แคช backfill) — ถ้าจะรัน audit ซ้ำต้องตั้ง env `MODE1600_TEAM_TASK_B`
- **ตรวจ SHA:** 730d ตรง `task_b/raw_data_intraday/README.md` ของทีม · snapshot ทุกไฟล์ตรง `sha256` ใน `.meta.json` · `mode1600/data1600.py` ตรวจซ้ำทุกครั้งที่โหลด
- ค่าเริ่มต้นของ `mode1600/data1600.py` อ่านจากโฟลเดอร์นี้ · env `MODE1600_TEAM_TASK_B` ยังชี้ไป clone ของทีมได้ (ทางเลือก)
- โหมด 16:00 มี **`mode1600/LOCK_1600.md` แยกของตัวเอง** — ไม่แก้ `LOCK.md` ของ next_day

| ไฟล์ (ใต้ `data/intraday/`) | ไบต์ | หมายเหตุ | blob ทีม | SHA-256 (เต็ม) |
|---|---|---|---|---|
| `raw_data_intraday/ADVANC_BK_1h_730d.csv` | 312,741 | ไฟล์ล็อก 2023-10-02 → 2026-09-25 · SHA ตรง README ทีม | `ad1bc278d222` | `2097bb53ce1c6ad49023f432128b39363187f4fdaebcb12bbd5ee7e5a295e79c` |
| `raw_data_intraday/KBANK_BK_1h_730d.csv` | 314,225 | ไฟล์ล็อก 2023-10-02 → 2026-09-25 · SHA ตรง README ทีม | `0ba968ed34a3` | `b192ad2c0bc3ae2a75b732752369c56b7dd27d043900eb6e8fb2d5fb08912060` |
| `raw_data_intraday_live/20260930/ADVANC_BK_1h_162242.csv` | 25,313 | ทาง (ii) · ดาวน์โหลด 2026-09-30T16:22:42+07:00 | `c4b3d957b6b8` | `d7b95acc21cf8942d392e5d6524b61de30f4ae2039ae50b662a279409d517ccd` |
| `raw_data_intraday_live/20260930/ADVANC_BK_1h_162242.csv.meta.json` | 243 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `6430c7cfb629` | `b3aa2d321fdcf20106abfaada77f1bfc955fb3a7417a7a1dc8b76a100ef3b3ca` |
| `raw_data_intraday_live/20260930/KBANK_BK_1h_162242.csv` | 25,390 | ทาง (ii) · ดาวน์โหลด 2026-09-30T16:22:42+07:00 | `055db80ea971` | `075e62a4dfcc69f8da6b8eee67c4635ce7ab7fb93fe94541e157c891356b746b` |
| `raw_data_intraday_live/20260930/KBANK_BK_1h_162242.csv.meta.json` | 241 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `b2af9a012c4f` | `b1c7052f923bc3d8c3ebbde42c9ce51a4bd41a4e0e93d3cdd05aa995470b6841` |
| `raw_data_intraday_live/20261001/ADVANC_BK_1h_160222.csv` | 25,307 | ทาง (ii) · ดาวน์โหลด 2026-10-01T16:02:22+07:00 | `73f27530ceb5` | `53a6c27881d98ac7581dd62aa839d789735de3259e9d1587ef4ad0342bcc0852` |
| `raw_data_intraday_live/20261001/ADVANC_BK_1h_160222.csv.meta.json` | 243 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `34e72ee698df` | `f7a4d1af6784b59a286371bcd834d52107901a7d48d6eb135b72a848fdf32016` |
| `raw_data_intraday_live/20261001/KBANK_BK_1h_160222.csv` | 25,388 | ทาง (ii) · ดาวน์โหลด 2026-10-01T16:02:22+07:00 | `483d6f4a66b1` | `81b6b50832cd9b3a9243911956ee2655486ebf817d8306c88da85cf4b4fbf75b` |
| `raw_data_intraday_live/20261001/KBANK_BK_1h_160222.csv.meta.json` | 241 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `ab37d9acfdfa` | `e1d1552d91aead1cb4413d5e4ab7d37ff05d8e42f351e6bb8de756ad2838ca07` |
| `raw_data_intraday_live/20261002/ADVANC_BK_1h_160234.csv` | 25,371 | ทาง (ii) · ดาวน์โหลด 2026-10-02T16:02:34+07:00 | `d213d59185fa` | `5e275b0d9c9e12433cbf1b2043415960f2fbd3557fde4baf0c9376b2837abf4d` |
| `raw_data_intraday_live/20261002/ADVANC_BK_1h_160234.csv.meta.json` | 243 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `c84e5056a313` | `9fc8e627a5c9d6c289762713b0a3cbac11d6514f98975b0c4c309bad38ab7f1b` |
| `raw_data_intraday_live/20261002/KBANK_BK_1h_160234.csv` | 25,452 | ทาง (ii) · ดาวน์โหลด 2026-10-02T16:02:34+07:00 | `ca1958b9d737` | `9150cf09c28a0ef32028d4c8172ff3356505a368b4630a38736b8863877fde64` |
| `raw_data_intraday_live/20261002/KBANK_BK_1h_160234.csv.meta.json` | 241 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `e66b9fe8adf7` | `f45bb929745b27dc7bbdf5d85aa0724b08d06c29a2dd6f050ec1ff8d14d5b0ec` |
| `raw_data_intraday_live/20261005/ADVANC_BK_1h_160223.csv` | 25,369 | ทาง (ii) · ดาวน์โหลด 2026-10-05T16:02:23+07:00 | `27c8bd5ab430` | `8da2fa54e457baaa646bfd627a6a0392cea25a7cab59f0c8a9146152ccd35778` |
| `raw_data_intraday_live/20261005/ADVANC_BK_1h_160223.csv.meta.json` | 243 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `3130aacd26cb` | `868348acae8e4e26dbfe6c7257ea019eb1a6c14cc0e807ae474fe51cf7f2c5d9` |
| `raw_data_intraday_live/20261005/KBANK_BK_1h_160223.csv` | 25,454 | ทาง (ii) · ดาวน์โหลด 2026-10-05T16:02:23+07:00 | `5361563b87df` | `b2b1e95a3994b28d7ca146f7acf08fe9c74750f5767c49bb4e6daf70405e5f3f` |
| `raw_data_intraday_live/20261005/KBANK_BK_1h_160223.csv.meta.json` | 241 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `675413634d6a` | `cb5e5be48d8ad415af3be343f1de6ee91c95faaba9e51cd5add1b2cb187d558e` |
| `raw_data_intraday_live/20261012/ADVANC_BK_1h_160232.csv` | 26,711 | ทาง (ii) · ดาวน์โหลด 2026-10-12T16:02:32+07:00 | `84e263520a16` | `bbba6d122e2e76037a2dd79b7b2ab309f82ba5386153c61b34d7997cfadf69e5` |
| `raw_data_intraday_live/20261012/ADVANC_BK_1h_160232.csv.meta.json` | 243 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `a0049faab71a` | `5ea25fe5dc362493f1f64d501c100de9b3aef7f3f8d956aa762c31756e8732d3` |
| `raw_data_intraday_live/20261012/KBANK_BK_1h_160232.csv` | 26,804 | ทาง (ii) · ดาวน์โหลด 2026-10-12T16:02:32+07:00 | `ae6afef37863` | `589ac9c6e80542a6a21c68d3205b387c5ae8382d95cd4ef60fcd9959c9fa434a` |
| `raw_data_intraday_live/20261012/KBANK_BK_1h_160232.csv.meta.json` | 241 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `814b85e36f4c` | `afc9332ca8ac001cc612c647ff7b481ca4c9e2eca43b837552fedad516145114` |
| `raw_data_intraday_live/20261014/ADVANC_BK_1h_160257.csv` | 26,294 | ทาง (ii) · ดาวน์โหลด 2026-10-14T16:02:57+07:00 | `42bd8051f709` | `b1db12c011e8e33e7035113080fd60c664bd512116619965433279f74d83a223` |
| `raw_data_intraday_live/20261014/ADVANC_BK_1h_160257.csv.meta.json` | 237 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `c1f7ba44d4de` | `e566d25b25e44670e2f7ef5b1b0f773aed450eec408b8d8b599fb6a3774f956d` |
| `raw_data_intraday_live/20261014/KBANK_BK_1h_160257.csv` | 26,380 | ทาง (ii) · ดาวน์โหลด 2026-10-14T16:02:57+07:00 | `9ac1d228c4a3` | `c50b255c46b2c914ac9ee55df02a9ec63312d8f37d014a19ab8351d386b874ef` |
| `raw_data_intraday_live/20261014/KBANK_BK_1h_160257.csv.meta.json` | 235 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `d16afb605298` | `4262938f3d32fa7e226efb16cdab486efaf30acc73f524b18b9edaa556d1b187` |
| `raw_data_intraday_live/20261015/ADVANC_BK_1h_160310.csv` | 26,292 | ทาง (ii) · ดาวน์โหลด 2026-10-15T16:03:10+07:00 | `5c6c128bfc7b` | `da75b763e47ef7881d092b235270af431a6ce04e724b14c552029767cbb801f6` |
| `raw_data_intraday_live/20261015/ADVANC_BK_1h_160310.csv.meta.json` | 237 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `f7e33637230e` | `a6d0630fa5724d2af3a70667068502cfdfaf382456af98b8c1bd758d31b085b1` |
| `raw_data_intraday_live/20261015/KBANK_BK_1h_160310.csv` | 26,375 | ทาง (ii) · ดาวน์โหลด 2026-10-15T16:03:10+07:00 | `47a05a2c932b` | `b7e42ef460d7b7a178deec93905a0ac25ec4e008a2bc6cb3df426f2662a43d35` |
| `raw_data_intraday_live/20261015/KBANK_BK_1h_160310.csv.meta.json` | 235 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `e9ecc4d41f23` | `5fd5bd4c02b26a56f7ec93a0b1d2c633074e844ce15fe51a66348c4f2fc5ff32` |
| `raw_data_intraday_live/20261019/ADVANC_BK_1h_160311.csv` | 26,297 | ทาง (ii) · ดาวน์โหลด 2026-10-19T16:03:11+07:00 | `705103edcb6d` | `23fb3c5a026ec7ff213f90b5dc83cd7c613a587dec75b0f84927c33952758f1c` |
| `raw_data_intraday_live/20261019/ADVANC_BK_1h_160311.csv.meta.json` | 237 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `cba0d1bc1f1b` | `c2faa3ba13cb13be35e60a09b45e416d4400c0705694e4d131addfdd7247cfca` |
| `raw_data_intraday_live/20261019/KBANK_BK_1h_160311.csv` | 26,367 | ทาง (ii) · ดาวน์โหลด 2026-10-19T16:03:11+07:00 | `f86d39e84a28` | `929335815e1125f6c6c2aa08ceb86fde8008b76bad22e132396112ce85290bed` |
| `raw_data_intraday_live/20261019/KBANK_BK_1h_160311.csv.meta.json` | 235 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `75930325537d` | `c109f15525181ae1fc60161f0c52903ea2ef0251e4ceca9a0fd5d7fb1aab39ce` |
| `raw_data_intraday_live/20261020/ADVANC_BK_1h_160345.csv` | 26,292 | ทาง (ii) · ดาวน์โหลด 2026-10-20T16:03:45+07:00 | `393c69810113` | `10ed02eda6b57160037c59568eb3839b6457fa200340dca8f1bec9c5a3f6a62f` |
| `raw_data_intraday_live/20261020/ADVANC_BK_1h_160345.csv.meta.json` | 237 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `dc5726fd6447` | `666e6e877d99726900ed343fe3c1db11cea7df26abebd3ccb9d822b9f963f93c` |
| `raw_data_intraday_live/20261020/KBANK_BK_1h_160345.csv` | 26,359 | ทาง (ii) · ดาวน์โหลด 2026-10-20T16:03:45+07:00 | `08c56dec011f` | `7c2453bbbb0431d4461e0255d0397b38b8167bf84df260a4dfb98242d8b6cb01` |
| `raw_data_intraday_live/20261020/KBANK_BK_1h_160345.csv.meta.json` | 235 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `08bb66743c83` | `9e2dd0cc28033389fe351048f1fbb76871bd2d4d8d23926dc7ec2562fe50cba6` |
| `raw_data_intraday_live/20261022/ADVANC_BK_1h_160230.csv` | 26,299 | ทาง (ii) · ดาวน์โหลด 2026-10-22T16:02:30+07:00 | `7368d3db2d02` | `2114a1c7f3e21143b376b68852e1e9fe210601dd42b623bea7aa25778c892ed5` |
| `raw_data_intraday_live/20261022/ADVANC_BK_1h_160230.csv.meta.json` | 237 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `c1b42dd31bdc` | `919718b847c53a81e3242048c6d7f2b79ba8af234be9079a093980bad6335a01` |
| `raw_data_intraday_live/20261022/KBANK_BK_1h_160230.csv` | 26,359 | ทาง (ii) · ดาวน์โหลด 2026-10-22T16:02:30+07:00 | `5e807393b40e` | `83a71864606f6cfffee73d11936cecc4e133d5bc3744dbc7b3ed75ace902b0a5` |
| `raw_data_intraday_live/20261022/KBANK_BK_1h_160230.csv.meta.json` | 235 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `7e71f54ce678` | `ae25e1618616872448b786e2d7d7e4e2ca091ff4e9ccd7608100111b7ebf23cb` |
| `raw_data_intraday_live/20261023/ADVANC_BK_1h_035532.csv` | 26,781 | ทาง (i) · ดาวน์โหลด 2026-10-23T03:55:32+07:00 | `9b85d5dee989` | `4cc988b5e3dd206b46a5c3a21ee7c8f31c956aab0e63495212e865e8ad2a1742` |
| `raw_data_intraday_live/20261023/ADVANC_BK_1h_035532.csv.meta.json` | 243 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `947f31e8b1b9` | `ccfa5c6f772ec10db50894cf4f440f11b7fa24aa59f436d6f0e62b1b696f6037` |
| `raw_data_intraday_live/20261023/KBANK_BK_1h_035532.csv` | 26,841 | ทาง (i) · ดาวน์โหลด 2026-10-23T03:55:32+07:00 | `38ca07ee3718` | `c14033f107e261d934f7d5a959141955bb883a05ae5a2bd766ef57df18a0de23` |
| `raw_data_intraday_live/20261023/KBANK_BK_1h_035532.csv.meta.json` | 241 | meta (เวลาดาวน์โหลด + SHA ของ csv) | `380ca3405338` | `28169d5f6469a9cbfb79040f1c0c1af2f86b5e60a63f98e3cc57e1a8143e05dc` |
