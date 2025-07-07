import bpy 
import os
import sys
import addon_utils

addon_path = os.path.join(os.environ['APPDATA'], 'Blender', '3.4', 'scripts', 'addons')
if addon_path not in sys.path:
    sys.path.append(addon_path)

with open(os.path.expanduser("~/deploy_plugin_log.txt"), "w") as f:
    f.write("✅ deploy-plugin.py has been executed\n")



print("==== sys.path ====")
for p in sys.path:
    print(p)
print("==================")

print("==== addon_utils.paths() ====")
for p in addon_utils.paths():
    print(p)
print("=============================")

bpy.ops.preferences.addon_enable(module='plugin')
bpy.ops.wm.save_userpref()
bpy.ops.script.reload()

