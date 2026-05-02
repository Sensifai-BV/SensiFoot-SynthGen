"""
feature_extractors.py
=====================
One function per ablation phase + baseline.
Each function receives a pandas DataFrame (already loaded from CSV)
and returns a float32 numpy array of shape (T, D).

Phase definitions (from paper):
  BASELINE  – identical to lstm_attention_slidingwindow.py         → 60 dims
  PHASE1    – 2D (X,Y) positions + temporal velocities             → 40 dims
  PHASE2    – 3D (X,Y,Z) positions + zero-norm joint angles
               + Savitzky-Golay filtered velocities                → 34 dims
  PHASE3    – PHASE2 feature set (SupCon loss handled in trainer)  → 34 dims
  PHASE4    – 3D Coords + knee angles + 2D bone lengths
               + absolute hip angles                               → 38 dims
  PHASE5    – 3D Coords + knee angles + 2D bone lengths
               (hip angles removed)                                → 36 dims
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── shared column list ──────────────────────────────────────────────────────
POS_COLS = [
    'L_Hip_x',      'L_Hip_y',      'L_Hip_z',
    'R_Hip_x',      'R_Hip_y',      'R_Hip_z',
    'L_Knee_x',     'L_Knee_y',     'L_Knee_z',
    'R_Knee_x',     'R_Knee_y',     'R_Knee_z',
    'L_Ankle_x',    'L_Ankle_y',    'L_Ankle_z',
    'R_Ankle_x',    'R_Ankle_y',    'R_Ankle_z',
    'L_Heel_x',     'L_Heel_y',     'L_Heel_z',
    'R_Heel_x',     'R_Heel_y',     'R_Heel_z',
    'L_Toe_x',      'L_Toe_y',      'L_Toe_z',
    'R_Toe_x',      'R_Toe_y',      'R_Toe_z',
    'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z',
    'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z',
]

def _load_pos(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(columns=POS_COLS, fill_value=0).apply(
        pd.to_numeric, errors='coerce').fillna(0)


def _torso_length(pos_df: pd.DataFrame) -> np.ndarray:
    s_mid_x = (pos_df['L_Shoulder_x'] + pos_df['R_Shoulder_x']) / 2.0
    s_mid_y = (pos_df['L_Shoulder_y'] + pos_df['R_Shoulder_y']) / 2.0
    s_mid_z = (pos_df['L_Shoulder_z'] + pos_df['R_Shoulder_z']) / 2.0
    h_mid_x = (pos_df['L_Hip_x'] + pos_df['R_Hip_x']) / 2.0
    h_mid_y = (pos_df['L_Hip_y'] + pos_df['R_Hip_y']) / 2.0
    h_mid_z = (pos_df['L_Hip_z'] + pos_df['R_Hip_z']) / 2.0
    tl = np.sqrt((s_mid_x - h_mid_x)**2 + (s_mid_y - h_mid_y)**2 +
                 (s_mid_z - h_mid_z)**2).values.reshape(-1, 1).astype(np.float32)
    tl[tl == 0] = 1e-6
    return tl


# ── helpers for angles ──────────────────────────────────────────────────────

def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex b for vectors b→a and b→c, shape (T,1)."""
    ba = a - b
    bc = c - b
    cos_angle = (np.einsum('ij,ij->i', ba, bc) /
                 (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8))
    return np.arccos(np.clip(cos_angle, -1.0, 1.0)).reshape(-1, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE  (60-dim)  – exact replica of lstm_attention_slidingwindow.py
# ══════════════════════════════════════════════════════════════════════════════
def extract_baseline(df: pd.DataFrame) -> np.ndarray:
    pos_df = _load_pos(df)
    torso_length = _torso_length(pos_df)

    rel_ankle_x = ((pos_df['L_Ankle_x'] - pos_df['R_Ankle_x'])
                   .values.reshape(-1, 1).astype(np.float32))
    rel_ankle_y = ((pos_df['L_Ankle_y'] - pos_df['R_Ankle_y'])
                   .values.reshape(-1, 1).astype(np.float32))
    rel_data = np.column_stack((rel_ankle_x, rel_ankle_y)).astype(np.float32)

    l_knee_lift      = (pos_df['L_Knee_y'] - pos_df['L_Hip_y']  ).values.reshape(-1,1).astype(np.float32)
    r_knee_lift      = (pos_df['R_Knee_y'] - pos_df['R_Hip_y']  ).values.reshape(-1,1).astype(np.float32)
    l_ankle_forward  = (pos_df['L_Ankle_z'] - pos_df['L_Hip_z'] ).values.reshape(-1,1).astype(np.float32)
    r_ankle_forward  = (pos_df['R_Ankle_z'] - pos_df['R_Hip_z'] ).values.reshape(-1,1).astype(np.float32)
    l_heel_backward  = (pos_df['L_Heel_z']  - pos_df['L_Hip_z'] ).values.reshape(-1,1).astype(np.float32)
    r_heel_backward  = (pos_df['R_Heel_z']  - pos_df['R_Hip_z'] ).values.reshape(-1,1).astype(np.float32)
    l_tibia_len      = (pos_df['L_Ankle_y'] - pos_df['L_Knee_y']).values.reshape(-1,1).astype(np.float32)
    r_tibia_len      = (pos_df['R_Ankle_y'] - pos_df['R_Knee_y']).values.reshape(-1,1).astype(np.float32)

    def _dist3(p1x,p1y,p1z, p2x,p2y,p2z):
        return np.sqrt((pos_df[p2x]-pos_df[p1x])**2 +
                       (pos_df[p2y]-pos_df[p1y])**2 +
                       (pos_df[p2z]-pos_df[p1z])**2).values.reshape(-1,1).astype(np.float32)

    l_hip_knee_dist   = _dist3('L_Hip_x','L_Hip_y','L_Hip_z','L_Knee_x','L_Knee_y','L_Knee_z')
    r_hip_knee_dist   = _dist3('R_Hip_x','R_Hip_y','R_Hip_z','R_Knee_x','R_Knee_y','R_Knee_z')
    l_knee_ankle_dist = _dist3('L_Knee_x','L_Knee_y','L_Knee_z','L_Ankle_x','L_Ankle_y','L_Ankle_z')
    r_knee_ankle_dist = _dist3('R_Knee_x','R_Knee_y','R_Knee_z','R_Ankle_x','R_Ankle_y','R_Ankle_z')

    l_leg_extension = (l_hip_knee_dist + l_knee_ankle_dist).astype(np.float32)
    r_leg_extension = (r_hip_knee_dist + r_knee_ankle_dist).astype(np.float32)

    gesture_data = np.hstack((
        l_knee_lift, r_knee_lift, l_ankle_forward, r_ankle_forward,
        l_heel_backward, r_heel_backward, l_tibia_len, r_tibia_len,
        rel_ankle_x, l_leg_extension, r_leg_extension
    ))

    inter_ankle_euclid = np.sqrt(rel_ankle_x**2 + rel_ankle_y**2).astype(np.float32)
    l_toe_heel_slope   = (pos_df['L_Toe_y'] - pos_df['L_Heel_y']).values.reshape(-1,1).astype(np.float32)
    r_toe_heel_slope   = (pos_df['R_Toe_y'] - pos_df['R_Heel_y']).values.reshape(-1,1).astype(np.float32)

    l_ankle_hip_x = (pos_df['L_Ankle_x'] - pos_df['L_Hip_x']).values.astype(np.float32)
    r_ankle_hip_x = (pos_df['R_Ankle_x'] - pos_df['R_Hip_x']).values.astype(np.float32)

    # NOTE: baseline uses OLD 3-dim new_spatial_data (inter_ankle_euclid 2D, not 3D)
    new_spatial_data = np.hstack((inter_ankle_euclid, l_toe_heel_slope, r_toe_heel_slope))

    pos_data = pos_df.iloc[:, :30].values.astype(np.float32)

    if len(pos_data) > 1:
        velocity_data     = np.diff(pos_data, axis=0)
        zero_pad          = np.zeros((1, pos_data.shape[1]), dtype=np.float32)
        velocity_data     = np.vstack((zero_pad, velocity_data))

        ankle_vel_data    = velocity_data[:, 12:18]
        smoothed_ankle_vel = pd.DataFrame(ankle_vel_data).rolling(window=5, min_periods=1).mean().values.astype(np.float32)

        accel_data        = np.diff(ankle_vel_data, axis=0)
        zero_pad_ankles   = np.zeros((1, 6), dtype=np.float32)
        accel_data        = np.vstack((zero_pad_ankles, accel_data))

        l_ankle_hip_x_vel = np.diff(l_ankle_hip_x).reshape(-1, 1)
        r_ankle_hip_x_vel = np.diff(r_ankle_hip_x).reshape(-1, 1)
        zero_pad_1d       = np.zeros((1, 1), dtype=np.float32)
        l_ankle_hip_x_vel = np.vstack((zero_pad_1d, l_ankle_hip_x_vel)).astype(np.float32)
        r_ankle_hip_x_vel = np.vstack((zero_pad_1d, r_ankle_hip_x_vel)).astype(np.float32)
    else:
        velocity_data      = np.zeros_like(pos_data)
        smoothed_ankle_vel = np.zeros((len(pos_data), 6),  dtype=np.float32)
        accel_data         = np.zeros((len(pos_data), 6),  dtype=np.float32)
        l_ankle_hip_x_vel  = np.zeros((len(pos_data), 1),  dtype=np.float32)
        r_ankle_hip_x_vel  = np.zeros((len(pos_data), 1),  dtype=np.float32)

    ankle_hip_vel_data = np.hstack((l_ankle_hip_x_vel, r_ankle_hip_x_vel))

    # 30 + 6 + 6 + 2 + 11 + 3 + 2 = 60
    return np.hstack((
        velocity_data,        # 30
        smoothed_ankle_vel,   #  6
        accel_data,           #  6
        rel_data,             #  2
        gesture_data,         # 11
        new_spatial_data,     #  3
        ankle_hip_vel_data    #  2
    ))


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1  (40-dim)  – 2D (X,Y) only + temporal velocities, NO Z-axis
# ══════════════════════════════════════════════════════════════════════════════
def extract_phase1(df: pd.DataFrame) -> np.ndarray:
    """
    2D (X, Y) Coordinates + Temporal Velocities.
    Z-axis is entirely removed to bypass depth jitter.
    40 dims = 20 XY positions * 2 (position + velocity).
    """
    pos_df = _load_pos(df)

    # Keep only the 20 X,Y pairs (skip Z)  → 20 columns
    xy_cols = []
    for joint in ['L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle',
                  'L_Heel', 'R_Heel', 'L_Toe', 'R_Toe']:
        xy_cols += [f'{joint}_x', f'{joint}_y']

    xy_data = pos_df.reindex(columns=xy_cols, fill_value=0).values.astype(np.float32)  # (T, 20)

    if len(xy_data) > 1:
        vel_data = np.diff(xy_data, axis=0)
        vel_data = np.vstack((np.zeros((1, xy_data.shape[1]), dtype=np.float32), vel_data))
    else:
        vel_data = np.zeros_like(xy_data)

    # 20 + 20 = 40
    return np.hstack((xy_data, vel_data)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2  (34-dim)  – 3D coords + zero-norm joint angles + SG-filtered vels
# ══════════════════════════════════════════════════════════════════════════════
def extract_phase2(df: pd.DataFrame) -> np.ndarray:
    """
    3D (X,Y,Z) Coordinates + Zero-Normalized Joint Angles
    + Savitzky-Golay Filtered Velocities.
    34 dims = 12 XYZ positions + 16 SG-filtered velocities + 6 joint angles
    Breakdown: 12 joint coords (hips/knees/ankles) + 16 SG vel + 6 angles = 34
    """
    pos_df = _load_pos(df)

    # 12 joint positions (hips, knees, ankles) → 12 cols
    coord_cols = [
        'L_Hip_x','L_Hip_y','L_Hip_z', 'R_Hip_x','R_Hip_y','R_Hip_z',
        'L_Knee_x','L_Knee_y','L_Knee_z','R_Knee_x','R_Knee_y','R_Knee_z',
    ]
    coord_data = pos_df.reindex(columns=coord_cols, fill_value=0).values.astype(np.float32)  # (T,12)

    # 16 SG-filtered velocities from 4 ankle + 4 knee XY (8 joints × 2 axes = 16)
    vel_cols = [
        'L_Ankle_x','L_Ankle_y','L_Ankle_z', 'R_Ankle_x','R_Ankle_y','R_Ankle_z',
        'L_Knee_x', 'L_Knee_y', 'L_Knee_z',  'R_Knee_x', 'R_Knee_y', 'R_Knee_z',
        'L_Hip_x',  'L_Hip_y',  'L_Hip_z',   'R_Hip_x',
    ]
    vel_raw = pos_df.reindex(columns=vel_cols, fill_value=0).values.astype(np.float32)

    if len(vel_raw) > 1:
        raw_vel = np.diff(vel_raw, axis=0)
        raw_vel = np.vstack((np.zeros((1, raw_vel.shape[1]), dtype=np.float32), raw_vel))
        # Savitzky-Golay: window_length must be odd and ≤ T
        wl = min(7, len(raw_vel) if len(raw_vel) % 2 == 1 else len(raw_vel) - 1)
        wl = max(wl, 3)
        sg_vel = savgol_filter(raw_vel, window_length=wl, polyorder=2, axis=0).astype(np.float32)
    else:
        sg_vel = np.zeros((len(vel_raw), 16), dtype=np.float32)

    # 6 zero-normalized joint angles: L/R knee flexion, L/R hip-knee, L/R ankle
    def xyz(joint):
        return pos_df[[f'{joint}_x', f'{joint}_y', f'{joint}_z']].values.astype(np.float32)

    l_knee_angle  = _angle_3pts(xyz('L_Hip'),   xyz('L_Knee'),  xyz('L_Ankle'))
    r_knee_angle  = _angle_3pts(xyz('R_Hip'),   xyz('R_Knee'),  xyz('R_Ankle'))
    l_hip_angle   = _angle_3pts(xyz('L_Knee'),  xyz('L_Hip'),   xyz('R_Hip'))
    r_hip_angle   = _angle_3pts(xyz('R_Knee'),  xyz('R_Hip'),   xyz('L_Hip'))
    l_ankle_angle = _angle_3pts(xyz('L_Knee'),  xyz('L_Ankle'), xyz('L_Heel'))
    r_ankle_angle = _angle_3pts(xyz('R_Knee'),  xyz('R_Ankle'), xyz('R_Heel'))

    # Zero-normalize angles (subtract mean per sequence)
    angles = np.hstack((l_knee_angle, r_knee_angle, l_hip_angle,
                        r_hip_angle, l_ankle_angle, r_ankle_angle))
    angles = (angles - angles.mean(axis=0, keepdims=True)).astype(np.float32)

    # 12 + 16 + 6 = 34
    return np.hstack((coord_data, sg_vel, angles))


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3  (34-dim)  – same features as Phase 2; SupCon loss used in trainer
# ══════════════════════════════════════════════════════════════════════════════
def extract_phase3(df: pd.DataFrame) -> np.ndarray:
    """Identical feature set to Phase 2; SupCon loss is handled in the trainer."""
    return extract_phase2(df)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4  (38-dim)  – 3D coords + knee angles + 2D bone lengths + hip angles
# ══════════════════════════════════════════════════════════════════════════════
def extract_phase4(df: pd.DataFrame) -> np.ndarray:
    """
    3D Coordinates (12) + Knee Angles (2) + 2D Bone Lengths (8)
    + Absolute Hip Angles (4) + Temporal Velocity Differences (12) = 38 dims
    Breakdown: 12 coords + 12 vel_diffs + 2 knee_angles + 8 bone_lengths_2d + 4 hip_angles = 38
    """
    pos_df = _load_pos(df)

    def xyz(j):
        return pos_df[[f'{j}_x', f'{j}_y', f'{j}_z']].values.astype(np.float32)

    def xy(j):
        return pos_df[[f'{j}_x', f'{j}_y']].values.astype(np.float32)

    # 12 3D joint coordinates (hips, knees, ankles)
    coord_cols = [
        'L_Hip_x','L_Hip_y','L_Hip_z','R_Hip_x','R_Hip_y','R_Hip_z',
        'L_Knee_x','L_Knee_y','L_Knee_z','R_Knee_x','R_Knee_y','R_Knee_z',
    ]
    coord_data = pos_df.reindex(columns=coord_cols, fill_value=0).values.astype(np.float32)  # (T,12)

    # 12 velocity diffs of those same coords
    if len(coord_data) > 1:
        vel_data = np.diff(coord_data, axis=0)
        vel_data = np.vstack((np.zeros((1, 12), dtype=np.float32), vel_data))
    else:
        vel_data = np.zeros_like(coord_data)

    # 2 knee flexion angles
    l_knee_angle = _angle_3pts(xyz('L_Hip'), xyz('L_Knee'), xyz('L_Ankle'))
    r_knee_angle = _angle_3pts(xyz('R_Hip'), xyz('R_Knee'), xyz('R_Ankle'))

    # 8 2D (XY-plane) bone lengths: femur L/R, tibia L/R, shin L/R, thigh height L/R
    def dist2d(j1, j2):
        d = xy(j1) - xy(j2)
        return np.linalg.norm(d, axis=1).reshape(-1, 1).astype(np.float32)

    l_femur_2d   = dist2d('L_Hip',   'L_Knee')
    r_femur_2d   = dist2d('R_Hip',   'R_Knee')
    l_tibia_2d   = dist2d('L_Knee',  'L_Ankle')
    r_tibia_2d   = dist2d('R_Knee',  'R_Ankle')
    l_heel_2d    = dist2d('L_Ankle', 'L_Heel')
    r_heel_2d    = dist2d('R_Ankle', 'R_Heel')
    l_toe_2d     = dist2d('L_Ankle', 'L_Toe')
    r_toe_2d     = dist2d('R_Ankle', 'R_Toe')

    bone_2d = np.hstack((l_femur_2d, r_femur_2d, l_tibia_2d, r_tibia_2d,
                         l_heel_2d, r_heel_2d, l_toe_2d, r_toe_2d))

    # 4 absolute hip angles (XZ and XY plane projections)
    l_hip_xy = _angle_3pts(xyz('L_Knee'), xyz('L_Hip'), np.column_stack([
        pos_df['L_Hip_x'].values + 1, pos_df['L_Hip_y'].values, pos_df['L_Hip_z'].values
    ]))
    r_hip_xy = _angle_3pts(xyz('R_Knee'), xyz('R_Hip'), np.column_stack([
        pos_df['R_Hip_x'].values + 1, pos_df['R_Hip_y'].values, pos_df['R_Hip_z'].values
    ]))
    l_hip_xz = _angle_3pts(xyz('L_Knee'), xyz('L_Hip'), np.column_stack([
        pos_df['L_Hip_x'].values + 1, pos_df['L_Hip_y'].values, pos_df['L_Hip_z'].values
    ]))
    r_hip_xz = _angle_3pts(xyz('R_Knee'), xyz('R_Hip'), np.column_stack([
        pos_df['R_Hip_x'].values + 1, pos_df['R_Hip_y'].values, pos_df['R_Hip_z'].values
    ]))

    hip_angles = np.hstack((l_hip_xy, r_hip_xy, l_hip_xz, r_hip_xz))

    # 12 + 12 + 2 + 8 + 4 = 38
    return np.hstack((coord_data, vel_data, l_knee_angle, r_knee_angle,
                      bone_2d, hip_angles)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5  (36-dim)  – Phase 4 minus hip angles (Gold Standard)
# ══════════════════════════════════════════════════════════════════════════════
def extract_phase5(df: pd.DataFrame) -> np.ndarray:
    """
    3D Coordinates (12) + Temporal Velocity Diffs (12) + Knee Angles (2)
    + 2D Bone Lengths (8) – Hip Angles removed.
    12 + 12 + 2 + 8 + 2 = 36 dims
    """
    pos_df = _load_pos(df)

    def xyz(j):
        return pos_df[[f'{j}_x', f'{j}_y', f'{j}_z']].values.astype(np.float32)

    def xy(j):
        return pos_df[[f'{j}_x', f'{j}_y']].values.astype(np.float32)

    coord_cols = [
        'L_Hip_x','L_Hip_y','L_Hip_z','R_Hip_x','R_Hip_y','R_Hip_z',
        'L_Knee_x','L_Knee_y','L_Knee_z','R_Knee_x','R_Knee_y','R_Knee_z',
    ]
    coord_data = pos_df.reindex(columns=coord_cols, fill_value=0).values.astype(np.float32)

    if len(coord_data) > 1:
        vel_data = np.diff(coord_data, axis=0)
        vel_data = np.vstack((np.zeros((1, 12), dtype=np.float32), vel_data))
    else:
        vel_data = np.zeros_like(coord_data)

    l_knee_angle = _angle_3pts(xyz('L_Hip'), xyz('L_Knee'), xyz('L_Ankle'))
    r_knee_angle = _angle_3pts(xyz('R_Hip'), xyz('R_Knee'), xyz('R_Ankle'))

    def dist2d(j1, j2):
        d = xy(j1) - xy(j2)
        return np.linalg.norm(d, axis=1).reshape(-1, 1).astype(np.float32)

    l_femur_2d = dist2d('L_Hip',   'L_Knee')
    r_femur_2d = dist2d('R_Hip',   'R_Knee')
    l_tibia_2d = dist2d('L_Knee',  'L_Ankle')
    r_tibia_2d = dist2d('R_Knee',  'R_Ankle')
    l_heel_2d  = dist2d('L_Ankle', 'L_Heel')
    r_heel_2d  = dist2d('R_Ankle', 'R_Heel')
    l_toe_2d   = dist2d('L_Ankle', 'L_Toe')
    r_toe_2d   = dist2d('R_Ankle', 'R_Toe')

    bone_2d = np.hstack((l_femur_2d, r_femur_2d, l_tibia_2d, r_tibia_2d,
                         l_heel_2d, r_heel_2d, l_toe_2d, r_toe_2d))

    # 12 + 12 + 2 + 8 + 2 ankle_lift (bonus depth cue, keeps 36)
    l_ankle_lift = (pos_df['L_Ankle_y'] - pos_df['L_Hip_y']).values.reshape(-1, 1).astype(np.float32)
    r_ankle_lift = (pos_df['R_Ankle_y'] - pos_df['R_Hip_y']).values.reshape(-1, 1).astype(np.float32)

    # 12 + 12 + 1 + 1 + 8 + 2 = 36
    return np.hstack((coord_data, vel_data, l_knee_angle, r_knee_angle,
                      bone_2d, l_ankle_lift, r_ankle_lift)).astype(np.float32)


# ── Registry ─────────────────────────────────────────────────────────────────
PHASE_REGISTRY = {
    'BASELINE': {'fn': extract_baseline, 'input_dim': 60, 'use_supcon': False},
    'PHASE1':   {'fn': extract_phase1,   'input_dim': 40, 'use_supcon': False},
    'PHASE2':   {'fn': extract_phase2,   'input_dim': 34, 'use_supcon': False},
    'PHASE3':   {'fn': extract_phase3,   'input_dim': 34, 'use_supcon': True },
    'PHASE4':   {'fn': extract_phase4,   'input_dim': 38, 'use_supcon': False},
    'PHASE5':   {'fn': extract_phase5,   'input_dim': 36, 'use_supcon': False},
    'STGCN':    {'fn': None,             'input_dim': 36, 'use_supcon': False},
    # NOTE: STGCN fn is set below after extract_stgcn is defined.
}


# ══════════════════════════════════════════════════════════════════════════════
#  ST-GCN feature extractor  (36-dim = 12 joints × 3 XYZ coords, raw)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Joint order (0-indexed) matches the adjacency matrix in models_zoo.py:
#   0 L_Hip    1 R_Hip    2 L_Knee   3 R_Knee
#   4 L_Ankle  5 R_Ankle  6 L_Heel   7 R_Heel
#   8 L_Toe    9 R_Toe   10 L_Shoulder 11 R_Shoulder
#
#  No feature engineering beyond mean-centering each sequence to remove
#  absolute position (the network must learn from relative motion / shape).
# ──────────────────────────────────────────────────────────────────────────────
_STGCN_JOINT_COLS = [
    'L_Hip_x',      'L_Hip_y',      'L_Hip_z',
    'R_Hip_x',      'R_Hip_y',      'R_Hip_z',
    'L_Knee_x',     'L_Knee_y',     'L_Knee_z',
    'R_Knee_x',     'R_Knee_y',     'R_Knee_z',
    'L_Ankle_x',    'L_Ankle_y',    'L_Ankle_z',
    'R_Ankle_x',    'R_Ankle_y',    'R_Ankle_z',
    'L_Heel_x',     'L_Heel_y',     'L_Heel_z',
    'R_Heel_x',     'R_Heel_y',     'R_Heel_z',
    'L_Toe_x',      'L_Toe_y',      'L_Toe_z',
    'R_Toe_x',      'R_Toe_y',      'R_Toe_z',
    'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z',
    'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z',
]


def extract_stgcn(df: pd.DataFrame) -> np.ndarray:
    """
    Raw 3D joint coordinates, mean-centred per sequence.
    Returns (T, 36) float32.
    The STGCN model internally reshapes this to (B, T, 12, 3).
    """
    data = (df.reindex(columns=_STGCN_JOINT_COLS, fill_value=0)
             .apply(pd.to_numeric, errors='coerce')
             .fillna(0)
             .values.astype(np.float32))          # (T, 36)

    # Mean-centre per axis across the whole sequence so absolute camera
    # position is removed and only relative joint layout remains.
    data = data - data.mean(axis=0, keepdims=True)
    return data


# Patch the registry now that extract_stgcn is defined
PHASE_REGISTRY['STGCN']['fn'] = extract_stgcn
