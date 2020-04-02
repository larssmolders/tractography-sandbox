from Solvers.initialvaluesolver import InitialValueSolver
import numpy as np

class SimpleStreamlineSolver(InitialValueSolver):
    def __init__(self, step_size, interpolator=None):
        super().__init__(interpolator)
        self.step_size = step_size

    def solve(self, x0):
        cur_pos = x0
        result = []
        while True:
            # observe derivative
            derivative = self.get_dataterm_at(cur_pos)

            # perform step
            cur_pos += self.step_size * derivative

            # append new position to result
            result.append(np.array(cur_pos))

            # check if we should stop
            if np.linalg.norm(derivative) < 1e-4:
                break

        return result
