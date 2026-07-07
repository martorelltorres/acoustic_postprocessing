#!/usr/bin/env python3

import copy
import numpy as np
import open3d as o3d
import time
import tf.transformations as tr


# =============================================================================
# FINAL MAP
# =============================================================================

def build_final_map(
        patches,
        pose_graph,
        voxel_size=0.2):

    final_map = o3d.geometry.PointCloud()

    for idx, patch in enumerate(patches):

        if idx >= len(pose_graph.nodes):
            break

        T = pose_graph.nodes[idx].pose

        transformed = copy.deepcopy(
            patch.pcd
        )

        transformed.transform(T)

        final_map += transformed

    if len(final_map.points) > 0:

        final_map = final_map.voxel_down_sample(
            voxel_size
        )

    return final_map


# =============================================================================
# TRAJECTORY EXTRACTION
# =============================================================================

def extract_trajectory(
        pose_graph):

    trajectory = []

    for node in pose_graph.nodes:

        T = node.pose

        pos = T[:3, 3]

        trajectory.append(pos)

    return np.array(trajectory)


# =============================================================================
# CREATE LINESET
# =============================================================================

def create_lineset(
        trajectory,
        color=[0, 0, 1]):

    if len(trajectory) < 2:
        return None

    lines = []

    for i in range(len(trajectory) - 1):

        lines.append([i, i + 1])

    if len(lines) == 0:
        return None

    line_set = o3d.geometry.LineSet()

    line_set.points = (

        o3d.utility.Vector3dVector(
            trajectory
        )
    )

    line_set.lines = (

        o3d.utility.Vector2iVector(
            lines
        )
    )

    colors = [color for _ in lines]

    line_set.colors = (

        o3d.utility.Vector3dVector(
            colors
        )
    )

    return line_set


# =============================================================================
# LOOP LINES
# =============================================================================

def create_loop_lines(
        pose_graph,
        color=[1, 0, 0]):

    if len(pose_graph.edges) == 0:
        return None

    node_positions = []

    for node in pose_graph.nodes:

        node_positions.append(
            node.pose[:3, 3]
        )

    node_positions = np.array(
        node_positions
    )

    lines = []

    for edge in pose_graph.edges:

        if edge.uncertain:

            lines.append([

                edge.source_node_id,

                edge.target_node_id
            ])

    if len(lines) == 0:
        return None

    loop_lines = o3d.geometry.LineSet()

    loop_lines.points = (

        o3d.utility.Vector3dVector(
            node_positions
        )
    )

    loop_lines.lines = (

        o3d.utility.Vector2iVector(
            lines
        )
    )

    colors = [color for _ in lines]

    loop_lines.colors = (

        o3d.utility.Vector3dVector(
            colors
        )
    )

    return loop_lines


# =============================================================================
# RECENT CLOUDS LAYER  (Option C / VIS-002)
# =============================================================================

def build_recent_clouds(
        patches,
        pose_graph,
        window=15,
        voxel=0.4):
    """
    Builds a single cloud with the last `window` patches transformed by
    their current graph pose. Each patch is tinted with a temporal gradient (from
    blue=oldest to red=most recent) so the order and where a misalignment
    might be are visible. `voxel` subsamples for performance.
    """

    n = min(len(patches), len(pose_graph.nodes))
    if n == 0:
        return None

    start = max(0, n - window)
    span = max(n - start, 1)

    merged = o3d.geometry.PointCloud()

    for k, idx in enumerate(range(start, n)):

        patch = patches[idx]
        if patch.pcd is None:
            continue

        cloud = copy.deepcopy(patch.pcd)
        cloud.transform(pose_graph.nodes[idx].pose)

        # Blue→red temporal gradient by age within the window.
        frac = k / span
        color = [frac, 0.15, 1.0 - frac]
        cloud.paint_uniform_color(color)

        merged += cloud

    if len(merged.points) == 0:
        return None

    if voxel is not None and voxel > 0:
        merged = merged.voxel_down_sample(voxel)

    return merged


class PoseGraphMonitor:

    def __init__(
            self,
            enabled=True,
            overview_zoom=0.18,
            min_zoom=0.03,
            overview_margin=2.5,
            show_clouds=True,
            cloud_window=15,
            cloud_voxel=0.4):

        self.vis = o3d.visualization.Visualizer()

        self.active = False

        self.overview_zoom = overview_zoom
        self.min_zoom = min_zoom
        self.overview_margin = overview_margin

        # --- Point cloud layer (Option C / VIS-002) ---
        # Shows the clouds of the last `cloud_window` patches transformed
        # by their current poses, to see the map being built and detect
        # alignment inconsistencies live. `cloud_voxel` subsamples so
        # the render does not choke the loop (None = no subsampling).
        self.show_clouds = show_clouds
        self.cloud_window = int(cloud_window)
        self.cloud_voxel = cloud_voxel
        self.cloud_geom = None

        if not enabled:
            return

        self.active = self.vis.create_window(
            window_name="Pose Graph Monitor",
            width=1600,
            height=900
        )

        if not self.active:
            return

        opt = self.vis.get_render_option()

        if opt is None:
            self.active = False
            return

        opt.background_color = np.array(
            [0.02, 0.02, 0.02]
        )

        opt.point_size = 5.0

        self.initialized = False

        self.traj_geom = None
        self.loop_geom = None
        self.node_geom = None

        self.ctr = self.vis.get_view_control()

    def update(self, pose_graph, patches=None):

        if not self.active:
            return

        if len(pose_graph.nodes) < 2:
            return

        trajectory = extract_trajectory(
            pose_graph
        )

        traj_lines = create_lineset(
            trajectory,
            color=[0, 1, 0]
        )

        loop_lines = create_loop_lines(
            pose_graph,
            color=[1, 0, 0]
        )

        # Cloud layer of the last N patches (if patches are provided and it is
        # enabled). Lets you see the map being built and detect
        # misalignments live.
        recent_clouds = None
        if self.show_clouds and patches is not None:
            recent_clouds = build_recent_clouds(
                patches,
                pose_graph,
                window=self.cloud_window,
                voxel=self.cloud_voxel
            )

        # =========================================================
        # NODE CLOUD
        # =========================================================

        node_cloud = o3d.geometry.PointCloud()

        node_cloud.points = (
            o3d.utility.Vector3dVector(
                trajectory
            )
        )

        colors = np.tile(
            np.array([[1, 1, 0]]),
            (len(trajectory), 1)
        )

        node_cloud.colors = (
            o3d.utility.Vector3dVector(
                colors
            )
        )

        # =========================================================
        # FIRST INITIALIZATION
        # =========================================================

        if not self.initialized:

            if traj_lines is not None:

                self.vis.add_geometry(
                    traj_lines,
                    reset_bounding_box=True
                )

                self.traj_geom = traj_lines

            if loop_lines is not None:

                self.vis.add_geometry(
                    loop_lines,
                    reset_bounding_box=False
                )

                self.loop_geom = loop_lines

            self.vis.add_geometry(
                node_cloud,
                reset_bounding_box=False
            )

            self.node_geom = node_cloud

            if recent_clouds is not None:
                self.vis.add_geometry(
                    recent_clouds,
                    reset_bounding_box=False
                )
                self.cloud_geom = recent_clouds

            self.initialized = True

        else:

            # =====================================================
            # REMOVE OLD
            # =====================================================

            if self.traj_geom is not None:
                self.vis.remove_geometry(
                    self.traj_geom,
                    reset_bounding_box=False
                )

            if self.loop_geom is not None:
                self.vis.remove_geometry(
                    self.loop_geom,
                    reset_bounding_box=False
                )

            if self.node_geom is not None:
                self.vis.remove_geometry(
                    self.node_geom,
                    reset_bounding_box=False
                )

            if self.cloud_geom is not None:
                self.vis.remove_geometry(
                    self.cloud_geom,
                    reset_bounding_box=False
                )
                self.cloud_geom = None

            # =====================================================
            # ADD UPDATED
            # =====================================================

            if traj_lines is not None:

                self.vis.add_geometry(
                    traj_lines,
                    reset_bounding_box=False
                )

                self.traj_geom = traj_lines

            if loop_lines is not None:

                self.vis.add_geometry(
                    loop_lines,
                    reset_bounding_box=False
                )

                self.loop_geom = loop_lines

            self.vis.add_geometry(
                node_cloud,
                reset_bounding_box=False
            )

            self.node_geom = node_cloud

            if recent_clouds is not None:
                self.vis.add_geometry(
                    recent_clouds,
                    reset_bounding_box=False
                )
                self.cloud_geom = recent_clouds

        # =========================================================
        # GLOBAL OVERVIEW CAMERA
        # =========================================================
        # Keeps a top-down view of the entire accumulated trajectory.
        # The zoom is intentionally kept far out to observe the
        # global evolution, not just the local zone of the last node.
        # =========================================================

        if len(trajectory) >= 2:

            ctr = self.vis.get_view_control()

            # Center of the full trajectory
            traj_center = trajectory.mean(axis=0)

            # Aim at the center of the trajectory
            ctr.set_lookat(traj_center.tolist())

            # Top-down view (from above, Z axis)
            ctr.set_up([0, 1, 0])
            ctr.set_front([0, 0, 1])

            xy_extent = np.ptp(
                trajectory[:, :2],
                axis=0
            )

            scene_extent = max(
                float(np.max(xy_extent)),
                1.0
            )

            zoom = self.overview_zoom / (
                1.0 + scene_extent / 100.0
            )

            zoom /= self.overview_margin

            zoom = max(
                self.min_zoom,
                min(self.overview_zoom, zoom)
            )

            ctr.set_zoom(zoom)

        self.vis.poll_events()
        self.vis.update_renderer()

        time.sleep(0.01)

    def close(self):

        if self.active:
            self.vis.destroy_window()
