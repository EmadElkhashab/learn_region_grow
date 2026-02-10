import os.path
import open3d as o3d
import numpy as np

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def pts_to_ply(filename):
    ply_file = os.path.join(os.path.dirname(filename), os.path.basename(filename)[:-4]) + ".ply"
    ply_file = open(ply_file, 'w')

    with open(filename, 'rb') as f:
        lines = 0
        for line in f:
            if lines == 0:
                num_points = line.decode("utf-8")

                #   Goofy code formatting because ply will include the whitespace if this is indented.
                ply_file.write(
                    """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar intensity
property uchar red
property uchar green
property uchar blue
end_header
""" % int(num_points)
                )

                lines += 1
                continue

            line = line.decode("utf-8")
            line = line.split()

            ply_file.write("%f %f %f %d %d %d %d\n"%(float(line[0]),float(line[1]),float(line[2]),int(line[3]),int(line[4]),int(line[5]),int(line[6])))
    
    ply_file.close()


def extract_xyzrgb_pts(filename):
    pts = []
    with open(filename, 'rb') as f:
        lines = 0
        for line in f:
            if lines == 0:
                lines += 1
                continue

            point = []

            line = line.decode("utf-8")
            line = line.split()

            for c in range(7):
                #   Skip luminance column.
                if c == 3:
                    continue

                point.append(float(line[c]))

            pts.append(point)

    pts = np.asarray(pts)
    cloud = o3d.utility.Vector3dVector(pts[:, :3])
    colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255)
    cloud = o3d.geometry.PointCloud(cloud)
    cloud.colors = colors
    
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(filename), os.path.basename(filename)[:-4]) + "_xyzrgb.pts", cloud)


def xyz_to_pts(filename):
    pts = []
    with open(filename, 'rb') as f:
        for line in f:
            point = []

            line = line.decode("utf-8")
            line = line.split()

            for c in range(9):
                #   Skip normal columns.
                if c in [0, 1, 2, 6, 7, 8]:
                    point.append(float(line[c]))

            pts.append(point)

    pts = np.asarray(pts)
    cloud = o3d.utility.Vector3dVector(pts[:, :3])
    colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255)
    cloud = o3d.geometry.PointCloud(cloud)
    cloud.colors = colors
    
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(filename), os.path.basename(filename)[:-4]) + ".pts", cloud)


def hue_to_rgb(p, q, t):
    #   Have to copy or else p values will be changed across the three calculations.
    channel = p.copy()

    idx = np.where(t < 0)[0]
    t[idx] += 1

    idx = np.where(t > 1)[0]
    t[idx] -= 1

    idx = np.where(t < 1/6)[0]
    channel[idx] = p[idx] + (q[idx] - p[idx]) * 6 * t[idx]

    idx = np.where(np.logical_and(t >= 1/6, t < 1/2))[0]
    channel[idx] = q[idx]

    idx = np.where(np.logical_and(t >= 1/2, t < 2/3))[0]
    channel[idx] = p[idx] + (q[idx] - p[idx]) * (2/3 - t[idx]) * 6

    return channel


#   Which scan should I segment for floor height?
def map_floor_height():
    floor = o3d.io.read_point_cloud(r"C:\Users\alongo6\Documents\scan_data\complete_seg_results\4x_channel_64ep_16seed_model\sev1_conference_rm-1_xyzrgb_seg_results_0.2\object_2.pts")
    # print(f"Num points: {np.asarray(floor.points).shape}")

    #   segment_plane uses RANSAC algorithm.
    #   Supply distance tolerance (distance removed from plane still considered on the plane),
    #   num_inliers specifies the initial subset of points to randomly sample to perform linear regression,
    #   num_iterations specifies the number of separate sampled linear regressions to run.
    _, indices = floor.segment_plane(0.3, 25, 5000)

    floor = floor.select_by_index(indices)

    #   Obtain y values (Would be z, but I saved a rotated version of floor, so now height is represented with y dim).
    heights = np.expand_dims(np.asarray(floor.points)[:, 1], axis=1)
    height_min, height_max = heights.min(), heights.max()

    #   Scale from 0 to 1 then multiple by .68 to chop off purple/red end of hue circle (68% is blue).
    #   This also pushes the lower end down closer to zero.  Does this bias the visualization?
    hue = 0.68 * ((heights - height_min) / ((height_max) - height_min))

    sat = np.full(hue.shape, 1, dtype=np.float32)
    light = np.full(hue.shape, 0.5, dtype=np.float32)

    #   q is 1, p is 0.
    q = (light + sat) - (light * sat)
    p = 2 * light - q

    colors = np.concatenate((hue_to_rgb(p, q, hue + 1/3), hue_to_rgb(p, q, hue), hue_to_rgb(p, q, hue - 1/3)), axis=1)
    floor.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([floor])

    # o3d.io.write_point_cloud(r"C:\Users\alongo6\Documents\scan_data\complete_seg_results\4x_channel_64ep_16seed_model\sev1_conference_rm-1_xyzrgb_seg_results_0.2\sev1_floor_height_heat_map.pts", floor)


#   For segmentation refinement.
class PointTracker():
    def __init__(self, cloud):
        self.init_color = None
        self.viz = None
        self.cloud = cloud
        self.redisplay = True


    #   Add some loop here that will update the window when colors have been changed.
    def render(self):
        while self.redisplay:
            self.viz = o3d.visualization.VisualizerWithVertexSelection()

            self.redisplay = False

            self.viz.create_window()

            self.viz.add_geometry(self.cloud)
            self.viz.update_renderer()

            self.viz.register_selection_changed_callback(self.record_selection_color)
            self.viz.run()

            self.viz.clear_geometries()
            self.viz.destroy_window()


    def record_selection_color(self):
        points = self.viz.get_picked_points()
        indices = [p.index for p in points]

        selection = np.asarray(self.cloud.colors)[indices]
        selection = np.unique(selection, axis=0)

        #   Ensure selection only contains one unique color.  Otherwise, an improper selection was performed.
        if self.init_color is None:
            if selection.shape[0] != 1:
                print("Invalid color selection.  Must include a single color.  Try again.")
            else:
                self.init_color = selection[0]
        else:
            #   Allow for multiple color selections by taking all unique selected colors.
            #   Then loop for each color and assign these points to init_color.
            colors = np.asarray(self.cloud.colors)

            for c in selection:
                if (c != self.init_color).all():
                    indices = np.where((colors == c).all(axis=1))[0]
                    colors[indices] = self.init_color

            self.cloud.colors = o3d.utility.Vector3dVector(colors)

            select = input("Enter Y to continue selection: ")
            
            if select.lower() != 'y':
                self.init_color = None
                self.redisplay = True
                self.viz.close()



def merge_segments(cloud):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    viz = o3d.visualization.VisualizerWithVertexSelection()

    tracker = PointTracker(cloud)
    tracker.render()

    return tracker.cloud