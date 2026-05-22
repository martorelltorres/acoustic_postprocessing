#!/usr/bin/env python3

import copy
import numpy as np
import open3d as o3d
import time


# =============================================================================
# FINAL MAP
# =============================================================================

def build_final_map(
        patches,
        pose_graph,
        voxel_size=0.2):

    final_map = o3d.geometry.PointCloud()

    for idx, patch in enumerate(patches):

        transformed = copy.deepcopy(
            patch.pcd
        )

        transformed.transform(
            pose_graph.nodes[idx].pose
        )

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


class PoseGraphMonitor:

    def __init__(self):

        self.vis = o3d.visualization.Visualizer()

        self.vis.create_window(
            window_name="Pose Graph Monitor",
            width=1600,
            height=900
        )

        opt = self.vis.get_render_option()

        opt.background_color = np.array(
            [0.02, 0.02, 0.02]
        )

        opt.point_size = 5.0

        self.initialized = False

        self.traj_geom = None
        self.loop_geom = None
        self.node_geom = None

        self.ctr = self.vis.get_view_control()

    def update(self, pose_graph):

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

        # =========================================================
        # AUTO CAMERA FOLLOW
        # =========================================================

        bbox = self.vis.get_view_control()

        self.vis.poll_events()
        self.vis.update_renderer()

        time.sleep(0.01)

    def close(self):

        self.vis.destroy_window()

