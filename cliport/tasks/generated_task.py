import numpy as np

from cliport.tasks import Task

from cliport.tasks.grippers import Spatula
import random

from cliport.utils import utils

import pybullet as p
import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p

class GeneratedTask(Task):
    """Build a car using blocks."""


    """Stack 5 cylinders of different colors (red, blue, green, yellow, and orange) on top of each other in a pyramid shape on a table."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "stack 5 cylinders of different colors (red, blue, green, yellow, and orange) on top of each other in a pyramid shape on a table"
        self.task_completed_desc = "done stacking cylinders."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add table.
        table_size = (0.4, 0.4, 0.02)
        table_urdf = 'table/table.urdf'
        table_pose = self.get_random_pose(env, table_size)
        env.add_object(table_urdf, table_pose, 'fixed')

        # Cylinder colors.
        colors = [utils.COLORS['red'], utils.COLORS['blue'], utils.COLORS['green'], utils.COLORS['yellow'], utils.COLORS['orange']]

        # Add cylinders.
        cylinder_size = (0.04, 0.04, 0.04)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinders = []
        for i in range(5):
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=colors[i])
            cylinders.append(cylinder_id)

        # Associate placement locations for goals.
        place_pos = [(0, -0.1, 0.02), (0, 0.1, 0.02), (0, -0.05, 0.06), (0, 0.05, 0.06), (0, 0, 0.1)]
        targs = [(utils.apply(cylinder_pose, pos), cylinder_pose[1]) for pos in place_pos]

        # Goal: cylinders are stacked in a pyramid shape.
        self.add_goal(objs=cylinders, matches=np.ones((5, 5)), targ_poses=targs, replace=False,
                rotations=True, metric='pose', params=None, step_max_reward=1, symmetries=[np.pi/2]*5,
                language_goal=self.lang_template)
