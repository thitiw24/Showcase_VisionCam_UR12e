# ระบบตรวจจับและหยิบวางบรรจุภัณฑ์อัตโนมัติด้วยกล้อง 3 มิติ และหุ่นยนต์อุตสาหกรรม
## (CPF Eye-in-Hand 3D Vision & Universal Robots UR12e Pick-and-Place System)
### ชื่อโครงการภายใน: `Knom_LastCode` (Production Master Release)

---

## 📌 ข้อมูลโครงการและหน่วยงาน (Project Information)

* **หน่วยงานผู้รับผิดชอบ:** ศูนย์วิศวกรรมกลาง หนองจอก  
  **บริษัท ซีพีเอฟ (ประเทศไทย) จำกัด (มหาชน)**  
  *CENTER OF ENGINEERING EXCELLENCE (Nong Chok) | CPF (THAILAND) PUBLIC COMPANY LIMITED*
* **ทีมงาน:** ทีมพัฒนาประสิทธิภาพและนวัตกรรม (Efficiency & Innovation Development Team)
* **ผู้จัดทำและส่งมอบระบบ:** นายธิติ วัลลิภากร (นักศึกษาฝึกงาน)
* **เวอร์ชันระบบ:** Version 3.0 (Unified Hardened Production & Handover Release)
* **วันที่ปรับปรุงล่าสุด:** ตุลาคม 2569

---

## 1. ภาพรวมของระบบ (System Overview)

โปรเจกต์นี้เป็นซอฟต์แวร์ควบคุมระบบตรวจจับและหยิบวางอัตโนมัติ (Automated 3D Vision-Guided Pick-and-Place) ทำงานร่วมกันระหว่าง:
1. **หุ่นยนต์อุตสาหกรรม Universal Robots UR12e:** เคลื่อนที่หยิบวางและควบคุมหัวดูดสุญญากาศ (Vacuum Gripper)
2. **กล้อง 3 มิติ Orbbec Gemini 2 (Eye-in-Hand):** ติดตั้งที่หน้าแปลนปลายแขนหุ่นยนต์ เชื่อมต่อผ่านพอร์ต **USB 3.0 พร้อมสายต่อขยาย (USB Extended Cable)** ถ่ายภาพสี RGB และสร้างแผนที่ความลึก (Depth Map) ความแม่นยำระดับมิลลิเมตร
3. **ระบบวิเคราะห์ภาพ AI YOLO26 (`best.pt`):** โมเดลตรวจจับและจำแนกคลาสชิ้นงาน (`CARTON` / `RedBox`, `BlueBox`, `Egg`) บนสายพานลำเลียงแบบเรียลไทม์
4. **คอมพิวเตอร์ประมวลผล (Host Vision PC):** รันหน้าจอควบคุม (GUI), เปิด Non-stop TCP Socket Server (Port 5000), สื่อสารกับหุ่นยนต์ผ่าน RTDE (Port 30004), และคำนวณแปลงพิกัด 3 มิติจากกล้องไปยังฐานหุ่นยนต์ (Robot Base Frame)

---

## 2. โครงสร้างโฟลเดอร์และไฟล์สำคัญ (Directory Structure)

```text
Knom_LastCode/
├── Data/                              # ไฟล์โมเดล AI และพารามิเตอร์การคาลิเบรต
│   ├── best.pt                        # โมเดล YOLO26 สำหรับตรวจจับจำแนกชิ้นงาน
│   ├── calibration.npz                # ค่า Intrinsic และ Distortion ของเลนส์กล้อง (จากการทำ Manual Calibration)
│   ├── T_cam_to_gripper.npz           # Hand-Eye Transformation Matrix (Camera -> Gripper)
│   ├── T_cam_to_gripper_manual.npy    # เมทริกซ์สำรองจากการวัดแมนนวล
│   └── T_cam_to_gripper_report.json   # รายงานผลสถิติความคลาดเคลื่อน Hand-Eye
│
├── ur_scripts/                        # สคริปต์ PolyScope สำหรับ UR12e (ฉบับปรับปรุงใหม่ล่าสุด)
│   ├── 00_init_variables.script       # กำหนดค่า IP, Port, ตัวแปรคลาส และ Secondary Z Guard
│   ├── 01_connection_ready.script     # เชื่อมต่อ TCP Socket และส่งสัญญาณ "READY"
│   ├── 02_send_start_and_class.script # สั่งเริ่มรอบใหม่ "START" และส่งชื่อคลาสที่ต้องการหยิบ
│   ├── 03_receive_target.script       # รอรับพิกัด X, Y, Z จาก PC (มี Timeout Guard 20 วินาที)
│   ├── 04_robot_motion.script         # ควบคุมการเคลื่อนที่และกริปเปอร์ดูดชิ้นงาน
│   ├── 05_send_done.script            # ส่งสัญญาณ "DONE" บันทึกจบแต่ละรอบ
│   ├── ur_complete_program.script     # สคริปต์ฉบับรวมขั้นตอนทั้งหมด
│   └── README_UR.md                   # คู่มือการจัด Program Tree บนหน้าจอ Teach Pendant
│
├── subtests/                          # ⭐ ชุดเครื่องมือทดสอบแยกส่วนและ Calibrate ระบบย่อย
│   ├── CALIBRATION_GUIDE.md           # ⭐ คู่มือขั้นตอนการทำ Calibration และแนวทาง Factory Intrinsic เชิงลึก
│   ├── README.md                      # สารบัญเครื่องมือและคำอธิบายสคริปต์ใน subtests
│   ├── check_camera.py                # ตรวจสอบการมองเห็นกล้อง Orbbec และ Serial Number
│   ├── test_robot_pose_rtde.py        # ทดสอบอ่าน Joint และ TCP Pose ของหุ่นยนต์ผ่าน RTDE
│   ├── preview_yolo_depth.py          # ทดสอบรัน AI พรีวิวภาพกล้องสดพร้อมแผนที่ความลึก
│   ├── test_yolo_images.py            # ทดสอบโมเดล YOLO กับไฟล์ภาพนิ่ง
│   ├── test_vision_manual_trigger.py  # ทดสอบกดปุ่ม G เพื่อจำลองการคำนวณและส่งพิกัด 3D
│   ├── calibrate_hand_eye.py          # โปรแกรมคำนวณ Hand-Eye Calibration (รองรับ Factory Intrinsic จากกล้องตรงๆ)
│   ├── calibrate_camera_intrinsic.py  # คาลิเบรตเลนส์กล้องด้วยแผ่น Chessboard (Manual Calibration)
│   ├── capture_camera_video.py        # บันทึกวิดีโอจากกล้องเพื่อเก็บ Dataset
│   ├── extract_video_frames.py        # แตกเฟรมภาพนิ่งจากวิดีโอเพื่อนำไป Label ข้อมูล
│   └── train_yolo_model.py            # สคริปต์คำสั่งสั่งเทรนโมเดล YOLO
│
├── logs/                              # แหล่งเก็บประวัติการทำงานหมุนเวียน (Universal Blackbox Logs)
│   └── blackbox_YYYYMMDD.log          # บันทึกเหตุการณ์ระดับ Milliseconds สำหรับวิเคราะห์ปัญหา
│
├── snapshots/                         # บันทึกหลักฐานภาพถ่ายเมื่อมีปัญหาหรือกด Snapshot (F12)
│   ├── snapshot_*_rgb.jpg             # ภาพสี RGB ขณะเกิดเหตุ
│   ├── snapshot_*_depth.png           # ภาพ Colormap แสดงระยะความลึก
│   └── snapshot_*_info.json           # ข้อมูลพิกัดและผลการตรวจจับทางคณิตศาสตร์
│
├── main.py                            # ⭐ โค้ดโปรแกรมหลัก (GUI Dark Theme + Fail-Closed Safety Engine)
├── utils.py                           # ฟังก์ชันแปลงภาพและประมวลผลข้อมูลความลึก (pyorbbecsdk2)
├── gui_config.json                    # ไฟล์ตั้งค่าค่าเริ่มต้นของโปรแกรม (ROI, Conf, Model Path)
├── cycle_log.csv                      # ไฟล์บันทึกสถิติรอบเวลาและพิกัดการหยิบจริงหน้างาน
├── requirements.txt                   # รายชื่อแพ็กเกจ Python ที่จำเป็น
├── run.bat                            # สคริปต์คลิกเดียวสำหรับเปิดโปรแกรม
├── install_autostart.bat              # สคริปต์ลงทะเบียนให้โปรแกรมเปิดอัตโนมัติเมื่อเปิดคอมพิวเตอร์
└── uninstall_autostart.bat            # สคริปต์ยกเลิกการเปิดโปรแกรมอัตโนมัติ
```

---

## 3. กลไกความปลอดภัยและความเสถียรระดับอุตสาหกรรม (Hardened Safety Engineering)

ในเวอร์ชันนี้ โค้ดได้รับการปรับปรุงเพื่อปิดช่องโหว่และป้องกันความเสียหายต่อเครื่องจักร 100%:

1. **Fail-Closed RTDE Binding:**  
   หากการสื่อสาร RTDE กับหุ่นยนต์ขาดหาย ฟังก์ชันอ่านพิกัดจะส่งคืนค่า `None` เสมอ (ยกเลิกการคืนค่า `np.eye(4)` ของเวอร์ชันทดลองเดิม) และบล็อกการส่งพิกัด เพื่อป้องกันไม่ให้แขนกลคำนวณพิกัดผิดทิศทางและพุ่งชนสายพาน
2. **3D Safety Envelope Interlock:**  
   มีตัวกรองตรวจสอบพิกัดปลายทางทั้ง 3 แกนก่อนส่งให้หุ่นยนต์ โดยเฉพาะแกนดิ่ง:
   $$Z \in [-60.0\text{ mm}, +80.0\text{ mm}]$$
   หากพิกัดหลุดจากกรอบความปลอดภัย ระบบจะปฏิเสธการส่งพิกัดทันที
3. **Frame Stall Timeout Watchdog (3.0s):**  
   เนื่องจากหน้างานมีการต่อสายกล้องผ่าน **USB Extended Cable** ซึ่งอาจเกิดการสะดุดของสัญญาณในบางจังหวะ ระบบมี Watchdog ตรวจจับ หากไม่มีเฟรมภาพสมบูรณ์ส่งมาเกิน 3 วินาที จะตัดการเชื่อมต่อและดึงระบบ Auto-Recovery ทำงานทันที
4. **Thread-Safe Camera Reconnect:**  
   ปุ่มกด *Cam Reconnect* บนหน้าจอทำงานผ่านการส่ง Flag ไปให้ Worker Thread หลักจัดการ ป้องกันปัญหาการเปิดกล้องซ้อนกัน 2 เธรด
5. **Quick Robot Port Probe:**  
   มีการยิงทดสอบพอร์ต 30004 แบบ Non-blocking (0.3s) ก่อนต่อ RTDE ป้องกันโปรแกรมค้าง 21 วินาทีในกรณีที่หุ่นยนต์ยังไม่ได้เปิดเครื่อง

---

## 4. คู่มือการกำหนดค่าระบบเครือข่ายและการแก้ไข IP Address (Network Configuration Guide)

ระบบใช้การสื่อสารแบบ Static IP ระหว่างคอมพิวเตอร์และหุ่นยนต์ หากในอนาคตต้องการย้ายสเตชัน หรือเปลี่ยนวง Network ของโรงงาน **ต้องแก้ไข IP ให้ตรงกันทั้งหมด 4 จุดหลัก** ดังนี้:

### จุดที่ 1: การตั้งค่าบนคอมพิวเตอร์ประมวลผล (Host Vision PC)
* เข้าไปที่ **Network Connections** ใน Windows -> คลิกขวาที่การ์ดแลน (Ethernet) -> **Properties** -> **Internet Protocol Version 4 (TCP/IPv4)**
* กำหนดค่า Static IP (ค่าเริ่มต้นปัจจุบัน):
  * **IP Address:** `10.42.5.50`
  * **Subnet Mask:** `255.255.255.0`
  * **Default Gateway:** เว้นว่างไว้ (หรือตามวง LAN โรงงาน)

### จุดที่ 2: โค้ดโปรแกรมหลักบน PC (`main.py`)
เปิดไฟล์ `main.py` แล้วแก้ไขตัวเลข IP ในส่วน **`ROBOT_SAFETY_CONFIG`** (บรรทัดที่ 124–128) และพอร์ต TCP Server (บรรทัดที่ 146):
```python
ROBOT_SAFETY_CONFIG = {
    ...
    # Strict IP Binding: กำหนด IP หุ่นยนต์และ PC ที่อนุญาตให้สื่อสาร
    "TARGET_ROBOT_IP": "10.42.5.151",   # <-- แก้เป็น IP ของหุ่นยนต์ UR12e
    "PC_IP": "10.42.5.50",              # <-- แก้เป็น IP ของเครื่อง PC นี้
    "RTDE_PORT": 30004,                 # พอร์ต RTDE มาตรฐานของ UR
    "RTDE_RETRY_INTERVAL_S": 3.0,
}

DEFAULT_TCP_PORT = 5000                 # <-- พอร์ต TCP Socket Handshake
```

### จุดที่ 3: สคริปต์หุ่นยนต์ UR บน PolyScope (`ur_scripts/00_init_variables.script`)
เปิดไฟล์ `ur_scripts/00_init_variables.script` แล้วแก้ไขบรรทัดที่ 6–7 ให้ชี้มายัง IP ของเครื่อง PC:
```urscript
pc_ip = "10.42.5.50"         # <-- แก้เป็น IP ของคอมพิวเตอร์ Vision PC
pc_port = 5000               # <-- แก้เป็น TCP Port ที่ตรงกับ main.py
```
*(และอย่าลืมไปตั้งค่า Static IP ของหุ่นยนต์ที่หน้าจอ Teach Pendant: **Settings -> System -> Network** ให้เป็น `10.42.5.151`)*

### จุดที่ 4: สคริปต์ทดสอบย่อยในโฟลเดอร์ `subtests/`
หากมีการทดสอบระบบย่อย ต้องแก้ไขตัวแปร `ROBOT_IP` ในไฟล์เหล่านี้ให้ตรงกัน:
* `subtests/test_robot_pose_rtde.py` (บรรทัดที่ 5: `ROBOT_IP = "10.42.5.151"`)
* `subtests/test_vision_manual_trigger.py` (บรรทัดที่ 49: `ROBOT_IP = "10.42.5.151"`)
* `subtests/calibrate_hand_eye.py` (บรรทัดที่ 39: `ROBOT_IP = "10.42.5.151"`)

---

## 5. ขั้นตอนการติดตั้งและเริ่มต้นใช้งาน (Getting Started)

### 5.1 ข้อกำหนดเบื้องต้น (Prerequisites)
* ระบบปฏิบัติการ Windows 10 หรือ Windows 11 (64-bit)
* Python 3.12 หรือ 3.13 (ต้องติ๊ก `Add python.exe to PATH` ขณะติดตั้ง)
* Microsoft Visual C++ 2015–2022 Redistributable (x64)

### 5.2 การติดตั้งไลบรารี
เปิด PowerShell ในโฟลเดอร์นี้ แล้วรันคำสั่ง:
```powershell
pip install -r requirements.txt
```

### 5.3 การเปิดใช้งานโปรแกรม
* **เปิดแบบปกติ:** ดับเบิลคลิกที่ไฟล์ `run.bat`
* **ตั้งค่าให้เปิดอัตโนมัติเมื่อเปิดเครื่องคอมพิวเตอร์:** คลิกขวาที่ `install_autostart.bat` แล้วเลือก **Run as administrator**

---

## 6. การใช้งานสคริปต์บนหุ่นยนต์ UR12e (`ur_scripts/`)

> ⚠️ **ข้อพึงระวังสำคัญ:**  
> สคริปต์ในโฟลเดอร์ `ur_scripts/` ของชุดนี้ **ใหม่และสะอาดกว่า (Cleaner & More Updated)** สคริปต์เวอร์ชันเดิมที่ค้างอยู่ใน Teach Pendant ของหุ่นยนต์จริง  
> ขอแนะนำให้นำไฟล์ `00_` ถึง `05_` หรือ `ur_complete_program.script` ไปอัปเดตลงในโปรแกรม PolyScope ของหุ่นยนต์ เพื่อให้ได้ระบบตัด Timeout 20 วินาที และ Secondary Z-Guard ที่ปลอดภัยสูงสุด (ดูรายละเอียดวิธีนำเข้าใน `ur_scripts/README_UR.md`)

---

## 7. การทดสอบและการสอบเทียบระบบย่อย (`subtests/`)

ก่อนเริ่มการผลิตจริง หรือเมื่อมีการตรวจเช็กความพร้อม ให้ทดสอบตามลำดับ:
1. `python subtests/check_camera.py` : ตรวจสอบว่าระบบมองเห็นกล้อง Orbbec และอ่านค่า Serial Number ได้ปกติ
2. `python subtests/test_robot_pose_rtde.py` : ตรวจสอบว่าอ่านค่าตำแหน่ง Joint และ TCP Pose ของหน้าแปลนหุ่นยนต์ได้
3. `python subtests/test_vision_manual_trigger.py` : ทดสอบตรวจจับชิ้นงานและกดปุ่ม 'G' เพื่อคำนวณพิกัด 3D ใน Robot Base Frame

### 💡 ข้อสังเกตสำคัญด้านการสอบเทียบ (Calibration Note):
* **แนะนำใช้ค่าโรงงาน (Factory Intrinsic):** ตัวกล้อง Orbbec Gemini 2 มีค่า Intrinsic จากโรงงานฝังอยู่ในชิปกล้องอยู่แล้ว หากมีการถอดเปลี่ยนหัวดูดหรือขยับตำแหน่งกล้อง **ให้รันเฉพาะ `calibrate_hand_eye.py` ได้เลย** (ระบบจะดึงค่าโรงงานให้อัตโนมัติ ไม่จำเป็นต้องใช้กระดานหมากรุกรัน `calibrate_camera_intrinsic.py` ใหม่)
* **การต่อยอด `main.py`:** ปัจจุบัน `main.py` ยังโหลดพารามิเตอร์เลนส์จาก `Data/calibration.npz` อยู่ หากต้องการปรับปรุงให้ `main.py` ดึง Factory Intrinsic ตรงๆ ดูขั้นตอนและตัวอย่างโค้ดโดยละเอียดได้ที่ **[subtests/CALIBRATION_GUIDE.md](subtests/CALIBRATION_GUIDE.md)**

---

## 8. ข้อมูลติดต่อและการส่งมอบงาน (Support & Handover)

หากต้องการศึกษาเอกสารคู่มือการใช้งานและการบำรุงรักษาฉบับเต็ม:
* **เอกสารคู่มือฉบับเต็ม:** ดูได้ที่ `USER_MANUAL_KNOM_BUABARN.pdf` หรือไฟล์คู่มือมาตรฐานของศูนย์วิศวกรรมกลางในโฟลเดอร์ `ManualUser/`
* **หน่วยงานต้นสังกัด:** ศูนย์วิศวกรรมกลาง หนองจอก, บริษัท ซีพีเอฟ (ประเทศไทย) จำกัด (มหาชน)
