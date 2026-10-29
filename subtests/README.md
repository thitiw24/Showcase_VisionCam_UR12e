# เครื่องมือและสคริปต์ทดสอบระบบย่อย (Subsystem Test & Utility Scripts)

โฟลเดอร์นี้รวบรวมเครื่องมือสำหรับทดสอบฮาร์ดแวร์ การตรวจจับ และการคาลิเบรตแต่ละส่วนแยกกัน เพื่อความสะดวกในการตรวจเช็กและแก้ไขปัญหา

---

## รายชื่อไฟล์และหน้าที่การทำงาน (Files & Functions)

### 1. หมวดทดสอบฮาร์ดแวร์และการเชื่อมต่อ (Hardware & Connection)
* **`check_camera.py`** (เดิม: `find_cam.py`)
  * ตรวจสอบว่าระบบมองเห็นกล้อง Orbbec หรือไม่ พร้อมแสดงชื่อรุ่นและ Serial Number
* **`test_robot_pose_rtde.py`** (เดิม: `test_ur_rtde.py`)
  * ทดสอบเชื่อมต่อกับหุ่นยนต์ UR ผ่าน RTDE และอ่านค่า Joint Positions และพิกัด Tool TCP Pose ปัจจุบัน

### 2. หมวดการมองเห็นและตรวจจับชิ้นงาน (Vision & YOLO Detection)
* **`preview_yolo_depth.py`** (เดิม: `CamDetectFusion.py`)
  * เปิดกล้องสดพร้อมรันโมเดล YOLO สามารถกดสลับโหมดดู Detection / Distance (ระยะลึก) / Object Size (ขนาดวัตถุ) ได้
* **`test_yolo_images.py`** (เดิม: `testmodel.py`)
  * ทดสอบนำโมเดล `best.pt` ไปทำนายภาพนิ่งในโฟลเดอร์เพื่อดูผลลัพธ์กรอบ Bounding Box
* **`test_vision_manual_trigger.py`** (เดิม: `final_test_01.py`)
  * เปิดกล้องสด ตรวจจับวัตถุ คำนวณพิกัด 3D Base ของหุ่นยนต์ และกดปุ่ม `G` บนคีย์บอร์ดเพื่อแสดงพิกัดที่จะส่งให้หุ่นยนต์ (โดยไม่ต้องเปิด TCP Server)

### 3. หมวดการคาลิเบรต (Calibration)
* **`calibrate_camera_intrinsic.py`** (เดิม: `Cam_Calb_Full02.py`)
  * ถ่ายภาพกระดานหมากรุก (Chessboard) เพื่อคำนวณค่าความบิดเบี้ยวและ Intrinsic Matrix ของเลนส์กล้อง
* **`calibrate_hand_eye.py`** (เดิม: `eye-in-hand-07.py`)
  * โปรแกรมเต็มสำหรับการทำ Eye-in-Hand Calibration เพื่อหาความสัมพันธ์ระหว่างกล้องกับกริปเปอร์หุ่นยนต์ (`T_cam_to_gripper.npz`)

### 4. หมวดเครื่องมือจัดการวิดีโอและเทรนโมเดล (Video & Training Tools)
* **`capture_camera_video.py`** (เดิม: `CamVideo.py`)
  * สคริปต์บันทึกภาพนิ่งหรืออัดคลิปวิดีโอจากกล้อง Orbbec เก็บไว้เป็นไฟล์
* **`extract_video_frames.py`** (เดิม: `VideoToJpg.py`)
  * ดึงเฟรมจากไฟล์วิดีโอ (.mp4) ออกมาเป็นภาพนิ่ง (.jpg) เพื่อนำไปทำ Dataset ติด Label
* **`train_yolo_model.py`** (เดิม: `yolo_train.py`)
  * สคริปต์สำหรับสั่งเทรนโมเดล YOLO เพิ่มเติมจาก Dataset

### 5. สคริปต์ดั้งเดิม (Legacy)
* **`legacy_cycle_detector.py`** (เดิม: `detect_to_robot_justBest.py`)
  * สคริปต์เวอร์ชันทดสอบเดิมที่รองรับคลาส `BLUE` ควบคู่กับ `CARTON` และ `EGG`
* **`utils.py`**
  * โมดูลแปลงข้อมูลเฟรมภาพและพิกเซลสำหรับ Orbbec SDK (ใช้ร่วมกับสคริปต์กล้องอื่นๆ)
