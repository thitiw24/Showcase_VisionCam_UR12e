import math
# import rtde_control
import rtde_receive

ROBOT_IP = "10.42.5.151"  # <-- your robot's actual IP

try:
    # rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    # --- Raw results (as before) ---
    joint_positions_rad = rtde_r.getActualQ()
    tcp_pose_m = rtde_r.getActualTCPPose()

    print("Current joint positions (rad):", joint_positions_rad)
    print("Current TCP pose (m):", tcp_pose_m)

    # --- Converted results ---
    joint_positions_deg = [math.degrees(q) for q in joint_positions_rad]

    tcp_pose_mm_deg = [
        tcp_pose_m[0] * 1000,          # x (mm)
        tcp_pose_m[1] * 1000,          # y (mm)
        tcp_pose_m[2] * 1000,          # z (mm)
        math.degrees(tcp_pose_m[3]),   # rx (deg)
        math.degrees(tcp_pose_m[4]),   # ry (deg)
        math.degrees(tcp_pose_m[5]),   # rz (deg)
    ]

    print("Current joint positions (deg):", [round(v, 3) for v in joint_positions_deg])
    print("Current TCP pose (mm/deg):", [round(v, 3) for v in tcp_pose_mm_deg])

except Exception as e:
    print(f"Connection failed: {e}")

finally:
    if 'rtde_c' in dir():
        rtde_c.disconnect()     # type: ignore[attr-defined]
    if 'rtde_r' in dir():
        rtde_r.disconnect()     # type: ignore[attr-defined]
    print("Disconnected.")