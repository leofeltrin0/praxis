import pybullet as p
import PySimpleGUI as sg
import pickle
from os import getcwd
from urdfpy import URDF
from os.path import abspath, dirname, basename, splitext
from transforms3d.affines import decompose
from transforms3d.quaternions import mat2quat
import numpy as np


class PyBulletRecorder:
    class LinkTracker:
        def __init__(self,
                     name,
                     body_id,
                     link_id,
                     link_origin,
                     mesh_path,
                     mesh_scale,
                     mesh_material=None):
            self.body_id = body_id
            self.link_id = link_id
            self.mesh_path = mesh_path
            self.mesh_scale = mesh_scale
            self.mesh_material = mesh_material
            decomposed_origin = decompose(link_origin)
            orn = mat2quat(decomposed_origin[1])
            orn = [orn[1], orn[2], orn[3], orn[0]]
            self.link_pose = [decomposed_origin[0],
                              orn]
            self.name = name

        def transform(self, position, orientation):
            return p.multiplyTransforms(
                position, orientation,
                self.link_pose[0], self.link_pose[1],
            )

        def get_keyframe(self):
            if self.link_id == -1:
                position, orientation = p.getBasePositionAndOrientation(
                    self.body_id)
                position, orientation = self.transform(
                    position=position, orientation=orientation)
            else:
                link_state = p.getLinkState(self.body_id,
                                            self.link_id,
                                            computeForwardKinematics=True)
                position, orientation = self.transform(
                    position=link_state[4],
                    orientation=link_state[5])
            return {
                'position': list(position),
                'orientation': list(orientation)
            }

    def __init__(self):
        self.states = []
        self.links = []

    def register_object(self, body_id, urdf_path, global_scaling=1, color=None):
        link_id_map = dict()
        n = p.getNumJoints(body_id)
        link_id_map[str(p.getBodyInfo(body_id)[0].decode('gb2312'))] = -1

        for link_id in range(0, n):
            link_id_map[str(p.getJointInfo(body_id, link_id)[
                12].decode('gb2312'))] = link_id

        dir_path = dirname(abspath(urdf_path))
        file_name = splitext(basename(urdf_path))[0]
        robot = URDF.load(urdf_path)
        for link in robot.links:
            # print("robot link:", body_id, link.name, link_id_map.keys())
            if link.name not in link_id_map:
                print("skip links !! ", link.name, link_id_map, len(robot.links), p.getBodyInfo(body_id)[0].decode('gb2312'))
                continue

            link_id = link_id_map[link.name]

            if len(link.visuals) > 0:
                for i, link_visual in enumerate(link.visuals):
                    mesh_material = None
                    if link_visual.material is not None:
                        mesh_material = link_visual.material
                        if color is not None:
                            mesh_material.name = mesh_material.name + f"_{np.random.randint(100)}" # mark it
                            mesh_material.color = color

                    if link_visual.geometry.mesh is not None:
                        print("use mesh", i, link_id_map.keys())

                        mesh_scale = [global_scaling,
                                      global_scaling, global_scaling]\
                            if link_visual.geometry.mesh.scale is None \
                            else link_visual.geometry.mesh.scale * global_scaling

                        self.links.append(('mesh',
                            PyBulletRecorder.LinkTracker(
                                name=file_name + f'_{body_id}_{link.name}_{i}',
                                body_id=body_id,
                                link_id=link_id,
                                link_origin=  # If link_id == -1 then is base link,
                                # PyBullet will return
                                # inertial_origin @ visual_origin,
                                # so need to undo that transform
                                (np.linalg.inv(link.inertial.origin)
                                 if link_id == -1
                                 else np.identity(4)) @
                                link_visual.origin * global_scaling,
                                mesh_path=dir_path + '/' +
                                link_visual.geometry.mesh.filename,
                                mesh_scale=mesh_scale,
                                mesh_material=mesh_material)))

                    if link_visual.geometry.box is not None:
                        print("use box", i, link_id_map.keys(), link_visual.geometry.box.__dict__)
                        # import IPython; IPython.embed()
                        mesh_scale =  link_visual.geometry.box.size / 2
                        self.links.append(('box',
                            PyBulletRecorder.LinkTracker(
                                name=file_name + f'_{body_id}_{link.name}_{i}',
                                body_id=body_id,
                                link_id=link_id,
                                link_origin= (np.linalg.inv(link.inertial.origin)
                                 if link_id == -1
                                 else np.identity(4)) @
                                link_visual.origin * global_scaling,
                                mesh_path='box',
                                mesh_scale=mesh_scale,
                                mesh_material=mesh_material)))


                    if link_visual.geometry.cylinder is not None:
                        print("use cylinder", i, link_id_map.keys(), link_visual.geometry.cylinder.__dict__)
                        mesh_scale = [link_visual.geometry.cylinder.radius, link_visual.geometry.cylinder.radius, link_visual.geometry.cylinder.length]
                        self.links.append(('cylinder',
                            PyBulletRecorder.LinkTracker(
                                name=file_name + f'_{body_id}_{link.name}_{i}',
                                body_id=body_id,
                                link_id=link_id,
                                link_origin= (np.linalg.inv(link.inertial.origin)
                                 if link_id == -1
                                 else np.identity(4)) @
                                link_visual.origin * global_scaling,
                                mesh_path='cylinder',
                                mesh_scale=mesh_scale,
                                mesh_material=mesh_material)))


                    if link_visual.geometry.sphere is not None:
                        print("use sphere", i, link_id_map.keys(), link_visual.geometry.sphere.__dict__)
                        mesh_scale = [link_visual.geometry.sphere.radius, link_visual.geometry.sphere.radius, link_visual.geometry.sphere.radius]
                        self.links.append(('sphere',
                            PyBulletRecorder.LinkTracker(
                                name=file_name + f'_{body_id}_{link.name}_{i}',
                                body_id=body_id,
                                link_id=link_id,
                                link_origin= (np.linalg.inv(link.inertial.origin)
                                 if link_id == -1
                                 else np.identity(4)) @
                                link_visual.origin * global_scaling,
                                mesh_path='sphere',
                                mesh_scale=mesh_scale,
                                mesh_material=mesh_material)))

    def add_keyframe(self):
        # Ideally, call every p.stepSimulation()
        current_state = {}
        for name, link in self.links:
            current_state[link.name] = link.get_keyframe()
        self.states.append(current_state)

    def prompt_save(self):
        layout = [[sg.Text('Do you want to save previous episode?')],
                  [sg.Button('Yes'), sg.Button('No')]]
        window = sg.Window('PyBullet Recorder', layout)
        save = False
        while True:
            event, values = window.read()
            if event in (None, 'No'):
                break
            elif event == 'Yes':
                save = True
                break
        window.close()

        if save:
            layout = [[sg.Text('Where do you want to save it?')],
                      [sg.Text('Path'), sg.InputText(getcwd())],
                      [sg.Button('OK')]]
            window = sg.Window('PyBullet Recorder', layout)
            event, values = window.read()
            window.close()
            self.save(values[0])
        self.reset()

    def reset(self):
        self.states = []

    def get_formatted_output(self):
        retval = {}
        for geo_name, link in self.links:
            if geo_name == 'mesh':
                retval[link.name] = {
                    'type': 'mesh',
                    'mesh_path': link.mesh_path,
                    'mesh_scale': link.mesh_scale,
                    'frames': [state[link.name] for state in self.states]
                }
            if geo_name == 'box':
                # print("retval: box!")
                retval[link.name] = {
                    'type': 'cube',
                    'name': link.name,
                    'mesh_scale': link.mesh_scale,
                    'frames': [state[link.name] for state in self.states]
                }
            if geo_name == 'cylinder':
                retval[link.name] = {
                    'type': 'cylinder',
                    'name': link.name,
                    'mesh_scale': link.mesh_scale,
                    'frames': [state[link.name] for state in self.states]
                }
            if geo_name == 'sphere':
                retval[link.name] = {
                    'type': 'sphere',
                    'name': link.name,
                    'mesh_scale': link.mesh_scale,
                    'frames': [state[link.name] for state in self.states]
                }
            if link.mesh_material is not None:
                retval[link.name]['mesh_material_name'] = link.mesh_material.name
                retval[link.name] ['mesh_material_color'] = link.mesh_material.color

        return retval

    def save(self, path):
        if path is None:
            print("[Recorder] Path is None.. not saving")
        else:
            print("[Recorder] Saving state to {}".format(path))
            pickle.dump(self.get_formatted_output(), open(path, 'wb'))
