import numpy as np
import open3d as o3d


def main():

    # read ply file
    pcd = o3d.io.read_point_cloud('pointCloud.ply')

    # visualize
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
