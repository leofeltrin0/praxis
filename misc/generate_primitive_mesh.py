import numpy as np
import trimesh

# generate unit length mesh to replace primitives

box = trimesh.creation.box(extents=[1, 1, 1])
trimesh.exchange.export.export_mesh(box, "box.obj")


cylinder = trimesh.creation.cylinder(radius=1, height=1)
trimesh.exchange.export.export_mesh(cylinder, "cylinder.obj")


sphere = trimesh.creation.icosphere()
trimesh.exchange.export.export_mesh(sphere, "sphere.obj")
