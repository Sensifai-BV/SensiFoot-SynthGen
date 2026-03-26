import bpy
import math
import mathutils
import os
import random
import sys
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Render FBX animation from multiple angles")
    parser.add_argument("--file_path", required=True, help="Path to input FBX file")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    return parser.parse_args(argv)

_args = parse_args()
FBX_FILE_PATH = _args.file_path
OUTPUT_DIR    = _args.output_dir

RESOLUTION_X  = 1080
RESOLUTION_Y  = 1080
FRAME_START   = 1
FRAME_END     = 220
BASE_FPS      = 30

# Camera geometry
BASE_DISTANCE  = 6.0
CAMERA_HEIGHT  = 1.3
LOOK_AT_HEIGHT = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# RENDER VARIANTS - UPDATED TO MATCH YOUR DRAWING
# ─────────────────────────────────────────────────────────────────────────────
SPEEDS           = [1.0]
DIST_MULTIPLIERS = [1.0]

# Mapping your drawing to Blender's coordinate degrees:
# F (Front) = 0°
# Teal 45 (Left) = -45°
# Blue 45 (Right) = 45°
# Pink 90 (Right) = 90°
ANGLE_OFFSETS = [
    ("Front",       0),
    ("FrontLeft", -45),
    ("FrontRight",  45),
    ("Right",       90),
]

TARGET_BONES = [
    "RightUpLeg", "RightLeg", "RightFoot",
    "LeftUpLeg",  "LeftLeg",  "LeftFoot",
    "Spine", "Spine1",
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# ROBUST FORWARD DETECTION: Symmetry Cross-Product
# ─────────────────────────────────────────────────────────────────────────────
def detect_true_forward(armature):
    """
    Bulletproof forward detection based on left/right body symmetry.
    Ignores arbitrary bone rolls and animation paths entirely.
    """
    scene = bpy.context.scene
    mat = armature.matrix_world
    bones = armature.pose.bones

    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()

    left_positions = []
    right_positions = []

    # 1. Parse bones into Left and Right sides based on standard naming conventions
    for b in bones:
        name = b.name.lower()
        if name.startswith("l_") or name.endswith("_l") or name.endswith(".l") or "left" in name:
            left_positions.append(mat @ b.head)
        elif name.startswith("r_") or name.endswith("_r") or name.endswith(".r") or "right" in name:
            right_positions.append(mat @ b.head)

    # 2. Calculate average left and right mass centers
    if len(left_positions) > 0 and len(right_positions) > 0:
        avg_left = sum(left_positions, mathutils.Vector()) / len(left_positions)
        avg_right = sum(right_positions, mathutils.Vector()) / len(right_positions)

        # 3. Create a vector pointing from Left to Right
        right_vec = (avg_right - avg_left)
        right_vec.z = 0  # Flatten to XY plane to ignore vertical posture imbalances
        
        if right_vec.length > 0.001:
            right_vec.normalize()
            up_vec = mathutils.Vector((0, 0, 1)) # Global Up (+Z)

            # 4. Cross Product (Up x Right = Forward)
            forward_vec = up_vec.cross(right_vec).normalized()
            forward_angle = math.atan2(forward_vec.y, forward_vec.x)
            
            print(f"[OK] Symmetry Detected! True Front Angle: {math.degrees(forward_angle):.2f}°")
            return forward_angle

    # FALLBACK: If rig has completely generic naming (no Left/Right indicators)
    print("[WARNING] No L/R symmetry found in bone names. Defaulting to 0°.")
    return 0.0

class KinematicAugmentor:
    def __init__(self, armature_obj, bones_list):
        self.armature  = armature_obj
        self.bones_list = bones_list
        self.modifiers  = []

    def apply_noise(self, intensity=1.0):
        action = self.armature.animation_data.action
        if not action: return
        for fcurve in action.fcurves:
            if 'pose.bones' not in fcurve.data_path or 'rotation' not in fcurve.data_path: continue
            bone_name = fcurve.data_path.split('"')[1]
            if bone_name in self.bones_list:
                mod = fcurve.modifiers.new(type='NOISE')
                self.modifiers.append((fcurve, mod))
                mod.phase = random.uniform(0, 100)
                mod.scale = random.uniform(10, 20)
                mod.strength = (0.08 * intensity * 0.5) if "Foot" in bone_name else (0.08 * intensity * random.uniform(0.8, 1.2))
                mod.blend_type = 'ADD'

    def clear_noise(self):
        for fcurve, mod in self.modifiers:
            fcurve.modifiers.remove(mod)
        self.modifiers = []

# ─────────────────────────────────────────────────────────────────────────────
# SCENE SETUP
# ─────────────────────────────────────────────────────────────────────────────
def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    imported = False
    for kwargs in [
        dict(automatic_bone_orientation=True, use_anim=True),
        dict(automatic_bone_orientation=True, use_anim=True, use_custom_normals=False),
    ]:
        try:
            bpy.ops.import_scene.fbx(filepath=FBX_FILE_PATH, **kwargs)
            imported = True
            break
        except Exception as e:
            pass
            
    if not imported:
        raise RuntimeError(f"Could not import FBX: {FBX_FILE_PATH}")

    armature = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)

    if bpy.context.scene.world:
        bpy.context.scene.world.use_nodes = True
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 1.5

    bpy.ops.object.light_add(type='AREA', location=(4, -4, 5))
    bpy.context.active_object.data.energy = 2000
    bpy.ops.object.light_add(type='AREA', location=(-4, -2, 3))
    bpy.context.active_object.data.energy = 800
    bpy.ops.object.light_add(type='AREA', location=(0, 6, 4))
    bpy.context.active_object.data.energy = 1200

    cam_data = bpy.data.cameras.new("Camera")
    cam_obj  = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    target.location = (0, 0, LOOK_AT_HEIGHT)

    ttc = cam_obj.constraints.new(type='TRACK_TO')
    ttc.target     = target
    ttc.track_axis = 'TRACK_NEGATIVE_Z'
    ttc.up_axis    = 'UP_Y'

    bpy.context.scene.render.resolution_x = RESOLUTION_X
    bpy.context.scene.render.resolution_y = RESOLUTION_Y
    bpy.context.scene.frame_start = FRAME_START

    frame_end = FRAME_END
    if armature and armature.animation_data and armature.animation_data.action:
        frame_end = int(armature.animation_data.action.frame_range[1])
    bpy.context.scene.frame_end = frame_end

    return cam_obj, armature

# ─────────────────────────────────────────────────────────────────────────────
# RENDER VARIANTS
# ─────────────────────────────────────────────────────────────────────────────
def render_variants(cam_obj, armature):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec  = 'H264'

    forward_angle = detect_true_forward(armature)
    augmentor = KinematicAugmentor(armature, TARGET_BONES)

    for speed in SPEEDS:
        scene.render.fps = int(BASE_FPS * speed)

        for dist_mult in DIST_MULTIPLIERS:
            current_dist = BASE_DISTANCE * dist_mult

            for angle_name, offset_deg in ANGLE_OFFSETS:
                augmentor.clear_noise()
                augmentor.apply_noise(intensity=1.0)

                # Camera placement based on the true symmetry angle
                cam_angle = forward_angle + math.radians(offset_deg)
                cam_pos = mathutils.Vector((
                    current_dist * math.cos(cam_angle),
                    current_dist * math.sin(cam_angle),
                    CAMERA_HEIGHT,
                ))
                cam_obj.location = cam_pos

                filename = f"speed{speed}_dist{dist_mult}_{angle_name}.mp4"
                scene.render.filepath = os.path.join(OUTPUT_DIR, filename)

                print(f"Rendering: speed={speed}x | dist={dist_mult}x | angle={angle_name} -> Offset: {offset_deg}°")
                bpy.context.view_layer.update()
                bpy.ops.render.render(animation=True)

    augmentor.clear_noise()
    print(f"\n[OK] Pipeline Render Finished -> {OUTPUT_DIR}\n")

if __name__ == "__main__":
    cam_obj, armature = setup_scene()
    if armature:
        render_variants(cam_obj, armature)
    else:
        print("No armature found — aborting.")