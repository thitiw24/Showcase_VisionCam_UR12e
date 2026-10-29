# คู่มือการสอบเทียบระบบวิชั่นและแนวทางการใช้ Factory Intrinsic
## (Camera & Hand-Eye Calibration Guide - `subtests/CALIBRATION_GUIDE.md`)

เอกสารฉบับนี้อธิบายขั้นตอนการสอบเทียบ (Calibration) สำหรับระบบวิชั่นหุ่นยนต์ Knom_LastCode ครอบคลุมการหาตำแหน่งกล้องกับกริปเปอร์ (Eye-in-Hand) และแนวทางการปรับปรุงระบบให้อ่านพารามิเตอร์เลนส์จากโรงงาน (Factory Intrinsic) โดยตรง

---

## 1. หลักการสำคัญ: ทำไมควรใช้ Factory Intrinsic มากกว่าการคาลิเบรตด้วยมือ?

* **ค่าจากโรงงาน (Factory Intrinsic):** กล้อง Orbbec Gemini 2 ได้รับการสอบเทียบจากห้องแล็บผู้ผลิตด้วยเครื่องมือความแม่นยำสูง และบันทึกค่าพารามิเตอร์ $f_x, f_y, c_x, c_y$ รวมถึงค่า Distortion ($k_1, k_2, p_1, p_2, k_3$) ลงในชิป EEPROM ของตัวกล้องเรียบร้อยแล้ว
* **การคาลิเบรตด้วยมือ (Manual Chessboard):** สคริปต์ `calibrate_camera_intrinsic.py` ทำงานโดยการให้ผู้ใช้ถือแผ่นกระดานหมากรุกส่ายหน้ากล้อง ซึ่งมีโอกาสเกิดความคลาดเคลื่อนจากระนาบแผ่นกระดานไม่เรียบ หรือแสงสะท้อน
* **ข้อสรุปเชิงปฏิบัติ:** **แนะนำให้ใช้ Factory Intrinsic เป็นค่าหลักเสมอ** โดยไม่ต้องรัน `calibrate_camera_intrinsic.py` ใหม่ เว้นแต่ชิปกล้องเสียหายหรือเปลี่ยนไปใช้กล้องรุ่นอื่นที่ไม่มีค่าบันทึกมา

---

## 2. ขั้นตอนการทำ Hand-Eye Calibration (`calibrate_hand_eye.py`)

เมื่อมีการถอดประกอบกริปเปอร์ ย้ายตำแหน่งยึดกล้อง หรือกริปเปอร์เกิดการชน ให้ทำการสอบเทียบ Hand-Eye ใหม่เพื่ออัปเดตไฟล์ `Data/T_cam_to_gripper.npz` ตามขั้นตอนดังนี้:

### 2.1 สิ่งที่ต้องเตรียม
1. แผ่นกระดานหมากรุก (Chessboard) ติดตั้งให้อยู่นิ่งสนิทบนพื้นโต๊ะ/สายพาน (ห้ามขยับเด็ดขาด)
   * **ค่าตั้งต้นในโปรแกรม (Default Parameters):** ขนาดตาราง $9 \times 7$ มุมใน (Inner corners) และขนาดช่องละ $20.0\text{ mm}$
   * **การปรับเปลี่ยนตามหน้างานจริง:** ขนาดตารางและขนาดช่องเป็นตัวแปรพารามิเตอร์ที่สามารถปรับเปลี่ยนได้ตามแผ่นกระดานจริงที่ใช้งานหน้างาน โดยเข้าไปแก้ไขตัวแปรที่ส่วนหัวของสคริปต์ (`calibrate_hand_eye.py` บรรทัดที่ 41–42 หรือ `calibrate_camera_intrinsic.py` บรรทัดที่ 38–39):
     ```python
     PATTERN_SIZE = (9, 7)      # จำนวนมุมในตาราง (Columns, Rows)
     SQUARE_SIZE_MM = 20.0      # ขนาดความกว้างของแต่ละช่องจริง (หน่วย mm)
     ```
2. เปิดคอนโทรลเลอร์หุ่นยนต์ UR12e และต่อสายแลนเข้ากับเครื่อง PC

### 2.2 ขั้นตอนการรัน
1. เปิด Terminal ในโฟลเดอร์โครงการ แล้วสั่งรัน:
   ```powershell
   python subtests/calibrate_hand_eye.py
   ```
2. **การโหลด Intrinsic อัตโนมัติ:** สคริปต์จะดึง Factory Intrinsic จากกล้อง Orbbec ให้โดยตรงทันที หากสำเร็จจะขึ้นข้อความ:
   ```text
   [CALIB] Factory Intrinsic successfully loaded directly from Orbbec camera
           fx=..., fy=..., cx=..., cy=...
   ```
   *(หากไม่เปิดกล้อง โปรแกรมจะถอยไปอ่านค่าสำรองจาก `Data/calibration.npz` อัตโนมัติ)*
3. ขยับแขนกล UR12e ไปยังตำแหน่งต่างๆ รอบแผ่นหมากรุก (อย่างน้อย 15 ท่าทาง) โดยปรับทั้งระยะ Translation ($X, Y, Z$) และมุมเอียงของข้อมือ (Rotation $R_x, R_y, R_z$)
4. ในแต่ละท่าทาง ให้รอจนแขนกลหยุดนิ่งสนิท แล้วกดปุ่ม `S` เพื่อบันทึกพิกัด
5. เมื่อครบตามจำนวน กด `Q` หรือคำนวณ โปรแกรมจะบันทึกผลลัพธ์ลงใน `Data/T_cam_to_gripper.npz` โดยตรงเพื่อให้ระบบหลักพร้อมทำงานทันที

---

## 3. สคริปต์ `calibrate_camera_intrinsic.py` (ใช้เมื่อใด?)

* ใช้เฉพาะกรณีฉุกเฉินเมื่อกล้องไม่ส่งค่า Factory Intrinsic หรือเปลี่ยนเลนส์ภายนอก
* คำสั่งรัน:
  ```powershell
  python subtests/calibrate_camera_intrinsic.py
  ```
* เล็งกล้องไปที่แผ่นกระดานหมากรุก กด `S` บันทึกภาพ 15–20 มุมมอง แล้วกด `Q` โปรแกรมจะสร้างไฟล์ `Data/calibration.npz`

---

## 4. แนวทางการต่อยอด: ปรับปรุง `main.py` ให้ใช้ Factory Intrinsic 100%

ปัจจุบัน `subtests/calibrate_hand_eye.py` รองรับ Factory Intrinsic แล้ว แต่ `main.py` ยังโหลดจากไฟล์ `Data/calibration.npz` อยู่ หากโปรแกรมเมอร์รุ่นถัดไปต้องการปลดการพึ่งพาไฟล์ `.npz` ให้ `main.py` ดึงค่าจากกล้องตรงๆ สามารถปรับโค้ดในฟังก์ชัน `_load_calibrations()` ของ `main.py` ได้ดังนี้:

### ตัวอย่างโค้ดปรับปรุงใน `main.py` (แนวทางสำหรับคนทำต่อ):
```python
def _load_calibrations(self):
    """Load intrinsic camera matrix (Factory first, fallback to npz) and hand-eye matrix."""
    try:
        # 1. พยายามอ่าน Factory Intrinsic จาก Pipeline กล้อง Orbbec โดยตรง
        if hasattr(self, 'camera_mgr') and self.camera_mgr.pipeline is not None:
            try:
                param = self.camera_mgr.pipeline.get_camera_param()
                rgb = param.rgb_intrinsic
                self.camera_matrix = np.array([
                    [rgb.fx, 0.0, rgb.cx],
                    [0.0, rgb.fy, rgb.cy],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                logger.info("[INIT] Loaded Factory Intrinsic directly from Orbbec camera.")
            except Exception as e:
                logger.warning(f"[INIT] Factory Intrinsic read failed, falling back: {e}")

        # 2. กรณีกล้องยังไม่ต่อ หรืออ่านไม่สำเร็จ ให้ถอยกลับมาใช้ไฟล์ calibration.npz
        if self.camera_matrix is None and CALIB_PATH.exists():
            calib = np.load(CALIB_PATH)
            self.camera_matrix = calib["camera_matrix"]
            logger.info("[INIT] Loaded Camera Intrinsic Matrix from calibration.npz.")

        # 3. โหลด Hand-Eye Matrix ตามปกติ
        if HANDEYE_PATH.exists():
            handeye = np.load(HANDEYE_PATH, allow_pickle=True)
            self.T_cam2gripper = handeye["T_cam_to_gripper"]
            logger.info("[INIT] Loaded Hand-Eye Matrix successfully.")
    except Exception as e:
        logger.error(f"[INIT ERROR] Error loading calibrations: {e}")
```

> **คำแนะนำ:** ตราบใดที่ยังไม่ได้ปรับแก้โค้ดข้างต้น ห้ามลบไฟล์ `Data/calibration.npz` ออกจากระบบเด็ดขาด
