import open3d as o3d
import numpy as np
from helpers.util import fit_shapes_to_box, params_to_8points, params_to_8points_no_rot
import json
import torch
from render.lineMesh import LineMesh
import pickle


def render(predBoxes, predAngles=None, classes=None, classed_idx=None, shapes_pred=None, render_type='points',
           render_shapes=True, render_boxes=False, colors=None):

    if render_type not in ['meshes', 'points']:
        raise ValueError('Render type needs to be either set to meshes or points.')

    if colors is None:
        colors = np.asarray(json.load(open('graphs/color_palette.json', 'r'))['rgb']) / 255.

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ren_opt = vis.get_render_option()
    # ren_opt.mesh_show_back_face = True
    ren_opt.line_width = 50.

    edges = [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]

    valid_idx = []
    all_pcl = []
    for i in range(len(predBoxes)-1):
        shape = shapes_pred[i]
        do_render_shape = True
        if render_type == 'points':
            vertices = shape
        else:
            do_render_shape = False
            if shape is not None:
                if len(shape) == 2:
                    vertices, faces = shape
                    did_fit = True
                else:
                    vertices, faces, did_fit = shape
                do_render_shape = True

        if classes[classed_idx[i]].split('\n')[0] in ["ceiling", "door", "doorframe"]:
            continue

        if predAngles is None:
            box_points = params_to_8points_no_rot(predBoxes[i])
        else:
            box_and_angle = torch.cat([predBoxes[i].float(), predAngles[i].float()])
            box_points = params_to_8points(box_and_angle, degrees=True)

        if do_render_shape:
            if predAngles is None:
                denorm_shape = fit_shapes_to_box(predBoxes[i], vertices, withangle=False)
            else:
                box_and_angle = torch.cat([predBoxes[i].float(), predAngles[i].float()])
                denorm_shape = fit_shapes_to_box(box_and_angle, vertices)

        valid_idx.append(i)
        if render_type == 'points':
            pcd_shape = o3d.geometry.PointCloud()
            pcd_shape.points = o3d.utility.Vector3dVector(denorm_shape)
            all_pcl += denorm_shape.tolist()
            pcd_shape_colors = [colors[i % len(colors)] for _ in range(len(denorm_shape))]
            pcd_shape.colors = o3d.utility.Vector3dVector(pcd_shape_colors)
            if render_shapes:
                vis.add_geometry(pcd_shape)
        else:
            mesh = o3d.geometry.TriangleMesh()
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.vertices = o3d.utility.Vector3dVector(denorm_shape)
            pcd_shape_colors = [colors[i % len(colors)] for _ in range(len(denorm_shape))]
            mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_shape_colors)

            if did_fit:
                mesh = mesh.subdivide_loop(number_of_iterations=1)
                mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
            mesh.compute_vertex_normals()
            if render_shapes:
                vis.add_geometry(mesh)

        if render_boxes:
            line_colors = [colors[i % len(colors)] for _ in range(len(edges))]
            line_mesh = LineMesh(box_points, edges, line_colors, radius=0.02)
            line_mesh_geoms = line_mesh.cylinder_segments

            for g in line_mesh_geoms:
                vis.add_geometry(g)

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 2]))
    vis.poll_events()
    vis.run()
    vis.destroy_window()

def render_off_screen(predBoxes, predAngles=None, scan_id=None, classes=None, classed_idx=None, shapes_pred=None,
           render_type='points', render_shapes=True, render_boxes=False,
           colors=None, output_folder=None):
    if render_type not in ['meshes', 'points']:
        raise ValueError('Render type needs to be either set to meshes or points.')

    if colors is None:
        colors = np.asarray(json.load(open('graphs/color_palette.json', 'r'))['rgb']) / 255.

    # Create offscreen renderer (adjust resolution as needed)
    width, height = 1920, 1080
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])  # White background

    # Choose a material based on the render type
    if render_type == 'points':
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
    else:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"

    geometry_id = 0
    for i in range(len(predBoxes) - 1):
        shape = shapes_pred[i]
        do_render_shape = True

        if render_type == 'points':
            vertices = shape
        else:
            do_render_shape = False
            if shape is not None:
                if len(shape) == 2:
                    vertices, faces = shape
                    did_fit = True
                else:
                    vertices, faces, did_fit = shape
                do_render_shape = True

        # Skip unwanted classes
        if classes[classed_idx[i]].split('\n')[0] in ["ceiling", "door", "doorframe", "wall"]:
            continue

        # Compute box corners
        if predAngles is None:
            box_points = params_to_8points_no_rot(predBoxes[i])
        else:
            box_and_angle = torch.cat([predBoxes[i].float(), predAngles[i].float()])
            box_points = params_to_8points(box_and_angle, degrees=True)

        # Fit the shape to the box
        if do_render_shape:
            if predAngles is None:
                denorm_shape = fit_shapes_to_box(predBoxes[i], vertices, withangle=False)
            else:
                box_and_angle = torch.cat([predBoxes[i].float(), predAngles[i].float()])
                denorm_shape = fit_shapes_to_box(box_and_angle, vertices)

        # Add the shape geometry
        if render_type == 'points':
            pcd_shape = o3d.geometry.PointCloud()
            pcd_shape.points = o3d.utility.Vector3dVector(denorm_shape)
            pcd_shape_colors = [colors[i % len(colors)] for _ in range(len(denorm_shape))]
            pcd_shape.colors = o3d.utility.Vector3dVector(pcd_shape_colors)
            if render_shapes:
                scene.add_geometry(f"pcd_{geometry_id}", pcd_shape, mat)
        else:
            mesh = o3d.geometry.TriangleMesh()
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.vertices = o3d.utility.Vector3dVector(denorm_shape)
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                [colors[i % len(colors)] for _ in range(len(denorm_shape))]
            )
            if did_fit:
                mesh = mesh.subdivide_loop(number_of_iterations=1)
                mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
            mesh.compute_vertex_normals()
            if render_shapes:
                scene.add_geometry(f"mesh_{geometry_id}", mesh, mat)

        # Optionally, add box edges as line geometries
        if render_boxes:
            lines = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5],
                              [2, 3], [2, 6], [3, 7], [4, 5], [4, 6],
                              [5, 7], [6, 7]])
            colors_lines = np.array([colors[i % len(colors)] for _ in range(len(lines))])
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(box_points),
                lines=o3d.utility.Vector2iVector(lines)
            )
            line_set.colors = o3d.utility.Vector3dVector(colors_lines)
            line_mat = o3d.visualization.rendering.MaterialRecord()
            line_mat.shader = "unlitLine"
            scene.add_geometry(f"lines_{geometry_id}", line_set, line_mat)
        geometry_id += 1

    # Add a coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 2])
    scene.add_geometry("coordinate_frame", coord_frame, mat)

    # Set up the camera. Adjust these parameters to get the desired view.
    fov = 45.0  # Field of view in degrees
    center = [0, 0, 0]   # Look-at center of the scene
    eye = [6, 6, 6]      # Camera position
    up = [0, 0, 1]       # Up vector
    renderer.setup_camera(fov, center, eye, up)

    # Render the scene to an image and save to disk
    img = renderer.render_to_image()
    o3d.io.write_image(output_folder+scan_id+'_view1.png', img)

    # Set up the camera. Adjust these parameters to get the desired view.
    fov = 45.0  # Field of view in degrees
    center = [0, 0, 0]   # Look-at center of the scene
    eye = [0, 0, 10]     # Camera position
    up = [0, 0, 1]       # Up vector
    renderer.setup_camera(fov, center, eye, up)

    # Render the scene to an image and save to disk
    img = renderer.render_to_image()
    o3d.io.write_image(output_folder+scan_id+'_view2.png', img)

    # Set up the camera. Adjust these parameters to get the desired view.
    fov = 45.0  # Field of view in degrees
    center = [0, 0, 0]   # Look-at center of the scene
    eye = [6, 6, -6]      # Camera position
    up = [0, 0, 1]       # Up vector
    renderer.setup_camera(fov, center, eye, up)

    # Render the scene to an image and save to disk
    img = renderer.render_to_image()
    o3d.io.write_image(output_folder+scan_id+'_view3.png', img)

if __name__ == "__main__":
    with open('/cluster/home/hanywu/scene_generation/graphto3d/render_inputs.pkl', 'rb') as f:
        print("ccccccccc")
        loaded_inputs = pickle.load(f)
        print("bbbbbbbbbbbbb")
        print(loaded_inputs.keys())
        render_off_screen(predBoxes=loaded_inputs['boxes_pred_den'], 
           predAngles=loaded_inputs['angles_pred'], 
           classes=loaded_inputs['classes'],
           classed_idx=loaded_inputs['classed_idx'],
           shapes_pred=loaded_inputs['shapes_pred'],
           render_type=loaded_inputs['render_type'],
           colors=loaded_inputs['colors'],
           render_boxes=True,)