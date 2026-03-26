"""
Dynamic Rokoko Retargeting Pipeline
Blender 4.0.2 | Linux | Headless (--background)

USAGE:
  cd /path/to/blender/directory
  ./blender --background --python retarget.py -- \
      --source  /path/to/deepmotion_anim.fbx     \
      --target  /path/to/mixamo_character.fbx    \
      --scheme  /path/to/deepmotion2mixamo.json  \
      --output  /path/to/output_directory

OPTIONAL FLAGS:
  --no-auto-scale       Disable auto-scaling (try if hips drift or feet clip)
  --pose CURRENT        Use current pose instead of rest pose (default: REST)
  --debug               Print full bone mapping table after build_bone_list
  --dry-run             Build bone list and print mapping, but don't retarget or export
"""

import bpy
import sys
import os
import argparse
import json


def parse_args():
    """Parse command line arguments."""
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(description="Rokoko headless retargeting pipeline")
    parser.add_argument("--source", required=True, help="Source (mocap) FBX")
    parser.add_argument("--target", required=True, help="Target (character) FBX")
    parser.add_argument("--scheme", required=True, help="Rokoko naming scheme JSON")
    parser.add_argument("--output", required=True, help="Output directory or file path")
    parser.add_argument("--no-auto-scale", action="store_true", help="Disable auto scaling")
    parser.add_argument("--pose", default="REST", choices=["REST", "CURRENT"])
    parser.add_argument("--debug", action="store_true", help="Print full bone list")
    parser.add_argument("--dry-run", action="store_true", help="Skip retarget + export")
    
    return parser.parse_args(argv)


def enable_rokoko():
    """Ensure the Rokoko addon is enabled in Blender."""
    import addon_utils
    addon_name = "rokoko-studio-live-blender-master"
    loaded, enabled = addon_utils.check(addon_name)
    
    if not enabled:
        result = addon_utils.enable(addon_name, default_set=False, persistent=False)
        if result is None:
            raise RuntimeError(
                f"Could not enable Rokoko addon '{addon_name}'.\n"
                f"Check folder name in: ~/.config/blender/4.0/scripts/addons/"
            )
    print(f"[OK] Rokoko addon enabled: '{addon_name}'")


def clear_scene():
    """Clear all objects, meshes, armatures, actions, and materials from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    for collection in [bpy.data.meshes, bpy.data.armatures, bpy.data.actions, bpy.data.materials]:
        for block in collection:
            collection.remove(block)
            
    print("[OK] Scene cleared")


def import_fbx(filepath):
    """
    Import FBX with progressive fallback mechanisms.
    Fixes Blender's headless binary FBX importer bug by toggling custom normal settings.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    before = {o.name for o in bpy.data.objects if o.type == 'ARMATURE'}

    # Attempt progressively more tolerant import configurations
    import_attempts = [
        dict(automatic_bone_orientation=True, use_anim=True, ignore_leaf_bones=False, force_connect_children=False),
        dict(automatic_bone_orientation=True, use_anim=True, ignore_leaf_bones=False, force_connect_children=False, use_custom_normals=False),
        dict(automatic_bone_orientation=True, use_anim=True, ignore_leaf_bones=True, force_connect_children=False, use_custom_normals=False, use_image_search=False),
    ]

    last_error = None
    for i, kwargs in enumerate(import_attempts, 1):
        try:
            bpy.ops.import_scene.fbx(filepath=filepath, **kwargs)
            last_error = None
            print(f"    [OK] Import succeeded on attempt {i} for {os.path.basename(filepath)}")
            break
        except Exception as e:
            print(f"    [!] Import attempt {i} failed: {e}")
            last_error = e
            # Clean up partial objects before retrying
            after_fail = {o.name for o in bpy.data.objects if o.type == 'ARMATURE'}
            for name in (after_fail - before):
                bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    if last_error:
        raise RuntimeError(
            f"All import attempts failed for '{filepath}'.\n"
            f"Last error: {last_error}\n"
            f"Try opening in Blender GUI -> File > Export > FBX -> re-save, then retry."
        )

    after = {o.name for o in bpy.data.objects if o.type == 'ARMATURE'}
    new_arms = after - before

    if not new_arms:
        raise RuntimeError(f"No armature found after importing '{filepath}'.")

    arm_name = sorted(new_arms)[0]
    arm = bpy.data.objects[arm_name]
    bone_names = sorted([b.name for b in arm.data.bones])
    
    print(f"    Armature : '{arm.name}'")
    print(f"    Bones ({len(bone_names)}): {bone_names[:8]}{'...' if len(bone_names) > 8 else ''}\n")

    return arm


def load_scheme(scheme_path):
    """Load the Rokoko naming scheme mapping."""
    scheme_path = os.path.abspath(scheme_path)
    if not os.path.exists(scheme_path):
        raise FileNotFoundError(f"Scheme JSON not found: {scheme_path}")

    result = bpy.ops.rsl.import_custom_schemes(filepath=scheme_path)
    if 'FINISHED' in result:
        print("[OK] Scheme loaded natively")
    else:
        print(f"    [!] import_custom_schemes returned {result} — applying manual injection")
        _inject_scheme_manually(scheme_path)

    with open(scheme_path) as f:
        data = json.load(f)
    print(f"    Canonical bones mapped: {len(data.get('bones', {}))}\n")


def _inject_scheme_manually(scheme_path):
    """Fallback logic to directly populate Rokoko's internal detection dict."""
    import importlib
    try:
        dm = importlib.import_module("rokoko-studio-live-blender-master.core.detection_manager")
    except ModuleNotFoundError:
        dm = importlib.import_module("rokoko_studio_live_blender_master.core.detection_manager")
        
    with open(scheme_path) as f:
        data = json.load(f)
        
    bones = data.get("bones", {})
    for canonical_key, name_list in bones.items():
        for bone_name in name_list:
            dm.bone_detection_list_custom[bone_name.lower()] = canonical_key


def retarget(source_arm, target_arm, args):
    """Configure and execute the Rokoko retargeting process."""
    scene = bpy.context.scene

    scene.rsl_retargeting_armature_source = source_arm
    scene.rsl_retargeting_armature_target = target_arm
    scene.rsl_retargeting_auto_scaling = not args.no_auto_scale

    if hasattr(scene, "rsl_retargeting_use_pose"):
        scene.rsl_retargeting_use_pose = args.pose

    result = bpy.ops.rsl.build_bone_list()
    if 'FINISHED' not in result:
        raise RuntimeError(f"build_bone_list() failed: {result}")

    bone_list = scene.rsl_retargeting_bone_list
    mapped = [b for b in bone_list if b.bone_name_source]
    unmapped = [b for b in bone_list if not b.bone_name_source]

    print(f"[OK] Bone list generated: {len(mapped)} mapped, {len(unmapped)} unmapped out of {len(bone_list)} total")

    if unmapped:
        print("    Unmapped target bones (no animation transferred):")
        for b in unmapped:
            print(f"        x {b.bone_name_target}")

    if args.debug:
        print("\n--- Full Bone Mapping ---")
        for b in sorted(bone_list, key=lambda x: x.bone_name_target):
            status = "OK" if b.bone_name_source else "XX"
            print(f"[{status}] {b.bone_name_target:40s} <- {b.bone_name_source or 'UNMAPPED'}")
        print("-------------------------\n")

    if not mapped:
        raise RuntimeError(
            "\n[ERR] ZERO bones were mapped!\n"
            "Bone names in your FBX files don't match any entry in the scheme JSON.\n"
            "Run with --debug to see actual bone names, then check your JSON."
        )

    if args.dry_run:
        print("[i] --dry-run active: skipping retargeting and export.")
        return False

    result = bpy.ops.rsl.retarget_animation()
    if 'FINISHED' not in result:
        raise RuntimeError(f"retarget_animation() failed: {result}")

    print("[OK] Retargeting complete")
    return True


def export_fbx(target_arm, output_path):
    """Export the retargeted target armature to FBX."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bpy.ops.object.select_all(action='DESELECT')
    target_arm.select_set(True)
    for child in target_arm.children:
        child.select_set(True)
    bpy.context.view_layer.objects.active = target_arm

    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        add_leaf_bones=False,
        axis_forward='-Z',
        axis_up='Y',
    )
    print(f"[OK] Exported successfully -> {output_path}")


def main():
    args = parse_args()

    # Generate the custom output filename 
    source_name = os.path.splitext(os.path.basename(args.source))[0]
    target_name = os.path.splitext(os.path.basename(args.target))[0]
    custom_filename = f"retarget_output_{source_name}_{target_name}.fbx"

    # Determine the output directory based on what the user passed
    if args.output.lower().endswith('.fbx'):
        output_dir = os.path.dirname(args.output)
    else:
        output_dir = args.output

    if not output_dir:
        output_dir = "."

    final_output_path = os.path.join(output_dir, custom_filename)

    print("\n" + "="*65)
    print(" Rokoko Headless Retargeting Pipeline")
    print(f" Source      : {args.source}")
    print(f" Target      : {args.target}")
    print(f" Scheme      : {args.scheme}")
    print(f" Output Dest : {final_output_path}")
    print(f" Settings    : AutoScale={not args.no_auto_scale} | Pose={args.pose} | DryRun={args.dry_run}")
    print("="*65 + "\n")

    enable_rokoko()
    clear_scene()
    load_scheme(args.scheme)
    
    source_arm = import_fbx(args.source)
    target_arm = import_fbx(args.target)

    if retarget(source_arm, target_arm, args):
        export_fbx(target_arm, final_output_path)
        print("\n[OK] Pipeline complete.\n")
    else:
        print("\n[OK] Dry run complete — inspect the bone mapping above.\n")


if __name__ == "__main__":
    main()
