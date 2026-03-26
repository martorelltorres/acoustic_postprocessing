#!/usr/bin/env python3

import rospy
import os
import open3d as o3d
import rasterio
import numpy as np
from std_msgs.msg import Bool
from matplotlib import cm

# Global flags to track completion
mb_finished = False
sss_finished = False

def mb_callback(msg):
    global mb_finished
    if msg.data:
        mb_finished = True
        rospy.loginfo("✔ Multibeam signal received.")

def sss_callback(msg):
    global sss_finished
    if msg.data:
        sss_finished = True
        rospy.loginfo("✔ Sidescan signal received.")

def main():
    rospy.init_node('fusion_node', anonymous=True)

    # -----------------------------
    # CONFIGURATION
    # -----------------------------
    mesh_file  = rospy.get_param('~mesh_file', '')
    sss_tif    = rospy.get_param('~sss_tif', '')
    output_dir = rospy.get_param('~output_dir', '.')
    
    if not mesh_file or not sss_tif:
        rospy.logerr("ERROR: Missing 'mesh_file' or 'sss_tif' parameters.")
        return

    output_mesh = os.path.join(output_dir, "mb_textured_sss.ply")
    COLORMAP = cm.gray
    NODATA_VALUE = 0

    # -----------------------------
    # ASYNCHRONOUS WAITING (THE FIX)
    # -----------------------------
    # Subscribe to both topics right away
    rospy.Subscriber('/pipeline/mb_done', Bool, mb_callback)
    rospy.Subscriber('/pipeline/sss_done', Bool, sss_callback)

    rospy.loginfo("Waiting for BOTH processes to finish...")
    
    rate = rospy.Rate(2) # Check 2 times per second
    while not (mb_finished and sss_finished):
        if rospy.is_shutdown():
            rospy.logwarn("Node interrupted while waiting.")
            return
        rate.sleep()

    rospy.loginfo("Both processes have finished! Proceeding to read files...")

    # -----------------------------
    # 1. LOAD MB MESH
    # -----------------------------
    rospy.loginfo(f"Loading MB mesh from: {mesh_file}")
    if not os.path.isfile(mesh_file):
        rospy.logerr(f"CRITICAL: Signal received, but the file {mesh_file} was not found.")
        return
        
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    rospy.loginfo(f"Vertices: {len(vertices)}")
    
    if len(vertices) == 0:
        rospy.logerr("ERROR: The loaded mesh has 0 vertices.")
        return

    # -----------------------------
    # 2. LOAD SSS MOSAIC
    # -----------------------------
    rospy.loginfo(f"Loading SSS mosaic from: {sss_tif}")
    if not os.path.isfile(sss_tif):
        rospy.logerr(f"CRITICAL: Signal received, but the file {sss_tif} was not found.")
        return

    try:
        with rasterio.open(sss_tif) as src:
            sss = src.read(1)
            transform = src.transform
            nodata = src.nodata

            # -----------------------------
            # 3. SSS → VERTICES PROJECTION
            # -----------------------------
            rospy.loginfo("Projecting SSS intensity onto the mesh...")
            intensity = np.zeros(len(vertices), dtype=np.float32)

            for i, (x, y, z) in enumerate(vertices):
                try:
                    row, col = src.index(x, y)
                    if 0 <= row < sss.shape[0] and 0 <= col < sss.shape[1]:
                        val = sss[row, col]
                        intensity[i] = NODATA_VALUE if (nodata is not None and val == nodata) else val
                    else:
                        intensity[i] = NODATA_VALUE
                except Exception:
                    intensity[i] = NODATA_VALUE

    except Exception as e:
        rospy.logerr(f"Failed to open SSS file: {e}")
        return

    # -----------------------------
    # 4. NORMALIZE + COLOR MAP
    # -----------------------------
    rospy.loginfo("Normalizing intensity and applying colormap...")
    valid = intensity > NODATA_VALUE
    rospy.loginfo(f"Vertices with valid intensity: {np.sum(valid)} / {len(valid)}")

    if not np.any(valid):
        rospy.logerr("ERROR: No mesh vertices intersect with the SSS mosaic.")
        return
        
    imin, imax = intensity[valid].min(), intensity[valid].max()
    int_norm = (intensity - imin) / (imax - imin + 1e-6)
    colors = COLORMAP(int_norm)[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # -----------------------------
    # 5. SAVE RESULT
    # -----------------------------
    rospy.loginfo(f"Saving textured mesh: {output_mesh}")
    o3d.io.write_triangle_mesh(output_mesh, mesh)
    rospy.loginfo("✅ Projection successfully completed!")

if __name__ == "__main__":
    main()