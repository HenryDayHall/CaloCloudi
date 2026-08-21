"""
Functions to test the timing, both in python and after compilation
It tests only the timing of the diffusion itself,
not the timing of casting back to physical coordinates and cells.
Thus, this is insensitive to actual physics output,
so they don't need to use trained models.
"""
import numpy as np
import time
import os
import yaml


class TimingData:
    def __init__(self, config, tag):
        self.config = config
        self.tag = tag
    
    def get_output_base(self):
        output_dir = self.config['output_dir']
        output_base_format = os.path.join(output_dir, f"times_{self.tag}_")
        i = 0
        n_zero_pad = 3
        padded_tail = str(i).zfill(n_zero_pad)
        while os.path.exists(output_base_format + padded_tail):
            i += 1
            padded_tail = str(i).zfill(n_zero_pad)
        output_base = output_base_format + padded_tail
        return output_base
    
    def save(self, cond, points, times):
        output_base = self.get_output_base()
        np.savez(output_base + ".npz", cond=cond, points=points, times=times)
        with open(output_base + ".yaml", "w") as f:
            yaml.dump(self.config, f)


def get_cond_and_n_points(config, n_events):
    """
    While the cond itself shouldn't effect the timing, the number of points per layer does
    as it changes the max points to be drawn.
    Take these from a dataset path specified by config.
    """
    cond = None # TODO
    points_per_layer = None
    n_points = np.sum(points_per_layer, axis=1)
    return cond, points_per_layer


def time_teacher_python(config, cond, points_per_layer):
    """
    Initialise the undistilled teacher, and time the inference on the
    given cond and points per layer. Timing should include the inference of the
    diffusion, not the conversion of the diffusion to physical coordinates and cells.
    """
    pass


def time_student_python(config, cond, points_per_layer):
    """
    Initialise the distilled student, and time the inference on the
    given cond and points per layer. Timing should include the inference of the
    diffusion, not the conversion of the diffusion to physical coordinates and cells.
    """
    pass


def time_compiled_student(config, cond, points_per_layer, compiled_path):
    """
    It's easier to give a compiled path, rather than create the compiled object.
    Then time the inference over the compiled object
    """
    pass
