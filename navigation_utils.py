from pymavlink import mavutil

def set_mode(mav, mode_name):
    if mav is None: return
    # Use pymavlink's built-in mapping
    mode_id = mav.mode_mapping().get(mode_name.upper())
    if mode_id is None: return
    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)

def send_waypoint(mav, lat, lon, alt):
    if mav is None: return
    mav.mav.mission_item_int_send(
        mav.target_system, mav.target_component, 0,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 2, 1, 0, 0, 0, 0,
        int(lat * 1e7), int(lon * 1e7), alt
    )

def set_roi(mav, lat, lon, alt=0):
    if mav is None: return
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_ROI_LOCATION, 0, 0, 0, 0, 0,
        lat, lon, alt
    )

def set_circle_radius(mav, radius):
    if mav is None: return
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        b'WP_LOITER_RAD', float(radius),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )

def send_loiter(mav, lat, lon, alt):
    if mav is None: return
    mav.mav.mission_item_int_send(
        mav.target_system, mav.target_component, 0,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM, 2, 1, 0, 0, 0, 0,
        int(lat * 1e7), int(lon * 1e7), alt
    )

def set_speed(mav, speed):
    if mav is None: return
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, 0, 1, speed, -1, 0, 0, 0, 0
    )
