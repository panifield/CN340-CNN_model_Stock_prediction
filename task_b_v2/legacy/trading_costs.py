"""
trading_costs.py — งาน B (feedback ข้อ 10: Breakeven / Commission)
===================================================================
ตอบคำถามเดียว: *สัญญาณที่โมเดลเห็น ใหญ่กว่าต้นทุนการซื้อขายหรือไม่*

*** diagnostic เท่านั้น -- ห้ามใช้เลือกโมเดล ***
การเลือกโมเดลใช้ MAE_return บน validation อย่างเดียว
ตารางนี้เป็นตารางที่สาม แยกจากตารางหลัก ไม่เข้า regression_metrics()

ไม่ใช้ trading_economics.py ของเดิม (มี Sharpe / t-stat / permutation
ซึ่งขัดกฎเหล็กข้อ 4 และคำนวณจาก prediction ของเวอร์ชันเก่า)
ห้ามเพิ่ม Sharpe / t-stat / p-value / equity curve / buy & hold ในเฟสนี้
"""

import numpy as np

from config import SET_COSTS, SET_TICK_BANDS


def cost_round_trip(costs=SET_COSTS):
    """ต้นทุนไป-กลับ (ซื้อ+ขาย) เป็นสัดส่วนของมูลค่าซื้อขาย"""
    per_side = (costs["commission_per_side"] + costs["trading_fee"]
                + costs["clearing_fee"] + costs["regulatory_fee"])
    return 2 * per_side * (1 + costs["vat"])


def tick_size(price, bands=SET_TICK_BANDS):
    """ขั้นราคาต่ำสุดของราคาระดับนั้น"""
    for lo, hi, tick in bands:
        if price >= lo and (hi is None or price < hi):
            return tick
    raise ValueError(f"ไม่พบ tick band ของราคา {price}")


def breakeven_report(y_true, y_pred, prev_close):
    """
    4 ตัวเลขที่ตอบ feedback ข้อ 10 -- diagnostic เท่านั้น ห้ามใช้เลือกโมเดล

    1. breakeven_bps
       ต้นทุนไป-กลับ = "กำไรขั้นต่ำที่ต้องได้จึงจะไม่ขาดทุน"

    2. pct_pred_above_cost  = mean(y_pred > cost_rt)          ** ทางเดียว ไม่ใช้ abs **
       % ของวันที่โมเดลทำนายว่า "ขึ้น" มากกว่าต้นทุน
       = สัดส่วนวันที่โมเดลเห็นโอกาส long ที่คุ้มค่าธรรมเนียม
       ถ้าเป็น 0% แปลว่าโมเดลไม่เคยแนะนำให้ซื้อเลยแม้แต่วันเดียว
       (ทำนายลงแรง ๆ ไม่นับ เพราะกลยุทธ์ short ไม่ได้)

    3. pct_actual_above_cost = mean(y_true > cost_rt)         ** ทางเดียว ไม่ใช้ abs **
       % ของวันที่ราคาขึ้นจริงมากกว่าต้นทุน
       = เพดานของกลยุทธ์ long/cash: ต่อให้โมเดลแม่นสมบูรณ์แบบ
         ก็มีวันที่ซื้อแล้วคุ้มได้ไม่เกินสัดส่วนนี้
       ตัวเลขนี้ไม่ขึ้นกับโมเดล -- เป็นคุณสมบัติของตลาด

    *** ข้อ 2-3 ต้องสอดคล้องกับ signal_economics() ที่เป็น long/cash ***

    4. mean_pred_move_ticks  (= mean_pred_move_baht / tick)
       ขนาดการเคลื่อนไหวที่โมเดลทำนาย |y_pred| * prev_close
       หารด้วย tick ของราคาวันนั้น แล้วเฉลี่ย (หารรายวันก่อนเฉลี่ย
       เพราะ tick เปลี่ยนตามระดับราคา)
       < 1 แปลว่าโมเดลทำนายการขยับที่เล็กกว่าที่ตลาดขยับได้จริง

    pct_* คืนเป็นสัดส่วน 0-1 (ไม่คูณ 100)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    prev = np.asarray(prev_close, dtype=float)
    cost_rt = cost_round_trip()

    ticks = np.array([tick_size(p) for p in prev])
    return {
        "breakeven_bps":         cost_rt * 1e4,
        "pct_pred_above_cost":   float(np.mean(y_pred > cost_rt)),
        "pct_actual_above_cost": float(np.mean(y_true > cost_rt)),
        "mean_pred_move_ticks":  float(np.mean(np.abs(y_pred) * prev / ticks)),
    }


def signal_economics(y_true, y_pred, cost_rt, threshold=None):
    """
    hypothetical close-to-close signal/cost diagnostic — not an executable backtest

    prediction ของวัน t ใช้ข้อมูลถึง close ของ t−1 และที่นี่สมมติว่าเข้า/ออกได้
    ที่ราคา close พอดี ซึ่งทำจริงไม่ได้ครบถ้วน (ไม่มี slippage / spread / ราคาจับคู่จริง)
    ตัวเลขจึงเป็นแค่ diagnostic ว่าสัญญาณใหญ่กว่าต้นทุนหรือไม่ ไม่ใช่ผลการเทรด

    สัญญาณแบบ long / cash เท่านั้น -- ไม่มี short

        position(t) = 1  ถ้า predicted_return(t) > threshold   (ซื้อถือ 1 วัน)
                      0  ถ้าไม่ใช่                             (ถือเงินสด)

    threshold (ประกาศล่วงหน้า ไม่จูน):
        default = cost_rt  -> "เข้าซื้อเฉพาะวันที่โมเดลคาดว่ากำไรเกินค่าธรรมเนียม"
                             ซึ่งเป็นนิยามของ breakeven ตรงตัว

    ทำไมไม่ short: short selling บน SET มีเงื่อนไขหุ้นที่ทำได้, ต้องยืมหุ้น,
    มีค่ายืม และกฎราคา -- การสมมติว่า short ได้ฟรีทุกวันจะทำให้ผลดีเกินจริง

    ต้นทุน: เริ่มและจบด้วยเงินสดเสมอ -- ถ้าวันสุดท้ายยังถือหุ้น ต้องนับค่าขายปิด position
    ผลตอบแทน: บวกกันแบบง่าย ไม่ทบต้น (อธิบายง่าย และ return รายวันเล็กมาก)

    n_trades = 0 คือผลลัพธ์ ไม่ใช่บั๊ก -- ห้ามลด threshold เพื่อให้มี trade

    คืน: n_days_long, n_trades, gross_return, total_cost, net_return
    (ห้ามเพิ่ม buy & hold หรือกลยุทธ์อ้างอิงอื่น -- กฎเหล็กข้อ 2 ห้ามเพิ่ม baseline)
    *** ห้ามเพิ่ม Sharpe / t-stat / p-value / equity curve ในเฟสนี้ ***
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if threshold is None:
        threshold = cost_rt

    position = (y_pred > threshold).astype(int)
    positions = np.r_[0, position, 0]
    turnover = np.abs(np.diff(positions))        # 1 = ซื้อหรือขายหนึ่งครั้ง
    total_cost = turnover.sum() * cost_rt / 2    # ครึ่ง round-trip ต่อครั้ง
    n_trades = int(turnover.sum() // 2)          # จำนวนรอบ ซื้อ->ขาย

    gross_return = float(np.sum(position * y_true))
    return {
        "n_days_long":  int(position.sum()),
        "n_trades":     n_trades,
        "gross_return": gross_return,
        "total_cost":   float(total_cost),
        "net_return":   gross_return - float(total_cost),
    }
