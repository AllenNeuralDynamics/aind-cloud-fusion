# TO BE DEPRECIATED
# May be useful for some geometry debugging. 

"""
Visualize Volumes
"""

from typing import List
import numpy as np
import plotly.graph_objects as go
import random

from aind_cloud_fusion import geometry

class Renderer:
    def __init__(self, volumes: List[geometry.Volume]): 
        self.volumes = volumes

    def render(self, see_vertices=True, see_normals=True):
        def get_random_rgb_string():
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            return f'rgb({r}, {g}, {b})'
        
        def create_go_mesh(vol: geometry.Volume, name: str, color: str): 
            verts = np.array(list(vol.vertices.values()))
            mesh = go.Mesh3d(x=verts[:, 0], 
                            y=verts[:, 1], 
                            z=verts[:, 2], 
                            flatshading=True, 
                            i=vol._simplices[:, 0], 
                            j=vol._simplices[:, 1], 
                            k=vol._simplices[:, 2], 
                            opacity=0.2,
                            name=name, 
                            color=color)
            return mesh

        def create_go_mesh_vertices(vol: geometry.Volume, color: str):
            verts = np.array(list(vol.vertices.values()))
            mesh_verts = go.Scatter3d(x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        mode='markers', 
                        marker=dict(color=color, size=1), 
                        showlegend=False)
            return mesh_verts

        def create_go_mesh_normals(vol: geometry.Volume, color: str):
            normals = []
            for face_vertex_ids, face_normal in zip(vol.faces.values(), vol.face_normals.values()):    
                face_vertex_ids = np.array(face_vertex_ids)
                verts = np.array(list(vol.vertices.values()))
                face_coordinates = verts[face_vertex_ids]
                face_centroid = np.average(face_coordinates, axis=0)

                start = face_centroid
                end = face_centroid + np.array(face_normal)

                normal = go.Scatter3d(x=[start[0], end[0]],
                                    y=[start[1], end[1]],
                                    z=[start[2], end[2]],
                                    mode='lines', 
                                    marker=dict(color=color, size=1), 
                                    showlegend=False)

                normals.append(normal)
            return normals

        # Rendering Code
        scene = []
        for i, vol in enumerate(self.volumes): 
            color = get_random_rgb_string()
            mesh = create_go_mesh(vol, f'Volume_{i}', color)
            scene.append(mesh)

            if see_vertices:
                mesh_vertices = create_go_mesh_vertices(vol, color)
                scene.append(mesh_vertices)
            if see_normals: 
                mesh_normals = create_go_mesh_normals(vol, color)
                scene.extend(mesh_normals)
                
        fig = go.Figure(data=scene)
        fig.show()