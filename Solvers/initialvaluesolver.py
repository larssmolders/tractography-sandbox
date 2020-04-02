

class InitialValueSolver(object):
    def __init__(self, interpolator=None):
        if type(self) is InitialValueSolver:
            raise Exception("InitialValueSolver is an abstract class and should not be initialized. "
                            "Use one of the derived classes (e.g. SimpleStreamlineSolver) instead")

        self.interpolator = interpolator

    def solve(self, x0):
        raise NotImplementedError("Class " + type(self).__name__ + " does not implement the solve() method")

    def get_dataterm_at(self, x):
        if self.interpolator is None:
            raise ValueError("Method get_dataterm_at is not implemented by subclass " + type(self).__name__ + " and no "
                                                                   "interpolating object was passed at initialization")
        else:
            return self.interpolator(x)
