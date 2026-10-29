# คู่มือการติดตั้งสคริปต์บนหุ่นยนต์ Universal Robots (UR PolyScope)

สคริปต์ในโฟลเดอร์นี้ถูกออกแบบมาเพื่อทำงานร่วมกับ `main.py` (หรือ `detect_to_robot_7.py`) โดยทำงานเป็นวัฏจักร (Cycle) อัตโนมัติ

---

## 1. โครงสร้างโปรแกรมทรีใน PolyScope (Program Tree Structure)

บนหน้าจอ PolyScope ให้สร้างโครงสร้างโปรแกรมดังนี้:

```
Robot Program
├── BeforeStart
│   └── Script: 00_init_variables.script          (กำหนด pc_ip, pc_port, target_class)
└── Loop (Always)
    ├── SubProgram / Script: 01_connection_ready.script   (เชื่อมต่อ TCP และส่ง READY)
    └── Loop (condition: is_connected)
        ├── SubProgram / Script: 02_send_start_and_class.script (ส่ง START และ CARTON/EGG)
        ├── SubProgram / Script: 03_receive_target.script       (รอรับพิกัด X,Y,Z จาก PC)
        ├── If (valid_target == True)
        │   ├── SubProgram / Script: 04_robot_motion.script     (เคลื่อนที่และควบคุม Gripper)
        │   └── SubProgram / Script: 05_send_done.script        (ส่ง DONE เพื่อจบและบันทึกรอบ)
        └── Wait: 0.2 s
```

---

## 2. ขั้นตอนการตั้งค่า IP Address

1. เปิดไฟล์ `00_init_variables.script`
2. แก้ไขค่า `pc_ip` ให้ตรงกับ IP ของเครื่องคอมพิวเตอร์ที่รันโปรแกรม Python เช่น:
   ```urscript
   pc_ip = "10.42.5.50"
   pc_port = 5000
   ```
3. กำหนดชิ้นงานเป้าหมายเริ่มต้นในตัวแปร `target_class`:
   * `"CARTON"` สำหรับกล่องไข่
   * `"EGG"` สำหรับไข่ไก่

---

## 3. ลำดับการทำงาน (Handshake Sequence)

1. **UR เชื่อมต่อ PC** -> ส่งข้อความ `"READY"`
2. **UR เริ่มต้นรอบใหม่** -> ส่งข้อความ `"START"`
3. **UR ระบุประเภทชิ้นงาน** -> ส่งข้อความ `"CARTON"` หรือ `"EGG"`
4. **PC ประมวลผล** -> ส่งพิกัดกลับมาในรูปแบบสตริง เช่น `"450.20,-201.40,153.70"` (หน่วย mm)
5. **UR เคลื่อนที่หยิบชิ้นงาน** ตามพิกัดที่แปลงเป็นเมตรแล้ว
6. **UR ส่งข้อความ `"DONE"`** -> PC จะคำนวณ Cycle Time บันทึกลง `cycle_log.csv` และรีเซ็ตรอบรอรับสัญญาณ `"START"` ถัดไป
