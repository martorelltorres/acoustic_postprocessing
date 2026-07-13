#!/usr/bin/env python3
"""
Texture the MB mesh with SSS intensity: geometry from multibeam, color from
the sidescan mosaic. Both share the UTM frame. Output: mb_textured_sss.ply.

Author: Antoni Martorell (SRV, UIB)
"""

import rospy
import os
import time
import open3d as o3d
import rasterio
import numpy as np
from std_msgs.msg import Bool
from matplotlib import cm

# Pipeline completion flags
mb_finished = False
sss_finished = False

def mb_callback(msg):
    global mb_finished
    if msg.data:
        mb_finished = True
        rospy.loginfo("Multibeam signal received.")

def sss_callback(msg):
    global sss_finished
    if msg.data:
        sss_finished = True
        rospy.loginfo("Sidescan signal received.")

def main():
    rospy.init_node('fusion_node', anonymous=True)

    mesh_file  = rospy.get_param('~mesh_file', '')
    sss_tif    = rospy.get_param('~sss_tif', '')
    output_dir = rospy.get_param('~output_dir', '.')
    
    if not mesh_file or not sss_tif:
        rospy.logerr("ERROR: Missing 'mesh_file' or 'sss_tif' parameters.")
        return

    mesh_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    output_mesh = os.path.join(mesh_dir, "mb_textured_sss.ply")
    COLORMAP = cm.gray
    NODATA_VALUE = 0
    # Color for vertices falling OUTSIDE the sidescan swath. A muted red rather than a
    # gray, which under cm.gray would be mistaken for real backscatter (and black for
    # low backscatter).
    NO_SSS_COLOR = (0.45, 0.12, 0.12)

    # Max seconds to wait for the upstream producers before falling back to
    # whatever already exists on disk. <=0 skips the wait and reads from disk.
    wait_timeout = rospy.get_param('~wait_timeout', 600.0)

    # Wait for both producers (timeout -> fall back to files on disk).
    # Uses wall-clock time so it works even without /clock (use_sim_time).
    rospy.Subscriber('/pipeline/mb_done', Bool, mb_callback)
    rospy.Subscriber('/pipeline/sss_done', Bool, sss_callback)

    # Checked before the loop: `wait_timeout > 0` used to live inside the break
    # condition, so wait_timeout=0 made the wait infinite instead of disabling it.
    if wait_timeout <= 0:
        rospy.loginfo("wait_timeout<=0: skipping signals, reading the files from disk.")
    else:
        rospy.loginfo("Waiting for BOTH processes to finish (timeout %.0fs)..." % wait_timeout)

        t_start = time.time()
        while not (mb_finished and sss_finished):
            if rospy.is_shutdown():
                rospy.logwarn("Node interrupted while waiting.")
                return
            if (time.time() - t_start) > wait_timeout:
                rospy.logwarn("Timeout waiting for upstream signals. "
                              "Using existing files on disk if available.")
                break
            time.sleep(0.5)

    rospy.loginfo("Proceeding to read files...")

    # Load MB mesh
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

    # Load SSS mosaic
    rospy.loginfo(f"Loading SSS mosaic from: {sss_tif}")
    if not os.path.isfile(sss_tif):
        rospy.logerr(f"CRITICAL: Signal received, but the file {sss_tif} was not found.")
        return

    try:
        with rasterio.open(sss_tif) as src:
            sss = src.read(1)
            nodata = src.nodata

            # VECTORIZED sampling of the SSS intensity at each vertex (x, y) in UTM.
            # src.index() accepts arrays; it used to be called once per vertex inside a
            # 1.5 M-iteration Python loop.
            rospy.loginfo("Projecting SSS intensity onto the mesh...")
            rows, cols = src.index(vertices[:, 0], vertices[:, 1])

    except Exception as e:
        rospy.logerr(f"Failed to open SSS file: {e}")
        return

    rows = np.asarray(rows)
    cols = np.asarray(cols)

    inside = (rows >= 0) & (rows < sss.shape[0]) & (cols >= 0) & (cols < sss.shape[1])

    intensity = np.full(len(vertices), NODATA_VALUE, dtype=np.float32)
    intensity[inside] = sss[rows[inside], cols[inside]]

    if nodata is not None:
        intensity[intensity == nodata] = NODATA_VALUE

    # Normalize and apply colormap to vertices
    rospy.loginfo("Normalizing intensity and applying colormap...")
    valid = intensity > NODATA_VALUE
    rospy.loginfo(
        f"Vertices with valid intensity: {np.sum(valid)} / {len(valid)} "
        f"({valid.mean() * 100:.1f}% over the SSS swath)"
    )

    if not np.any(valid):
        rospy.logerr("ERROR: No mesh vertices intersect with the SSS mosaic.")
        return

    # Normalize using ONLY the valid vertices. Normalizing the whole array maps the
    # no-data ones (intensity=0, below imin) to a NEGATIVE value that the colormap
    # saturates to black, making them indistinguishable from real low backscatter.
    imin, imax = intensity[valid].min(), intensity[valid].max()

    colors = np.tile(np.array(NO_SSS_COLOR), (len(vertices), 1))
    colors[valid] = COLORMAP((intensity[valid] - imin) / (imax - imin + 1e-6))[:, :3]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Save result
    rospy.loginfo(f"Saving textured mesh: {output_mesh}")
    o3d.io.write_triangle_mesh(output_mesh, mesh)
    rospy.loginfo("Projection completed.")

if __name__ == "__main__":
    main()