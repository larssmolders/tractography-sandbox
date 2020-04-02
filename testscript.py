import numpy as np
import dataloader as dl
from Solvers.simplestreamlinesolver import SimpleStreamlineSolver
from scipy.interpolate import RegularGridInterpolator
import renderer
import validator

fibercup_directions = dl.load_fibercup_tractography_derivatives_2D()

xmin = 0
xmax = len(fibercup_directions)
ymin = 0
ymax = len(fibercup_directions[0])
step_size = 1

x = np.arange(xmin, xmax, 1)
y = np.arange(ymin, ymax, 1)

directions = fibercup_directions[xmin:xmax, ymin:ymax]

interpolator = RegularGridInterpolator((x, y), directions)

streamline_solver = SimpleStreamlineSolver(step_size, interpolator=interpolator)

result = streamline_solver.solve([14.0, 39.0])

ground_truth = np.array([[14.0, 39.0], [30.0, 39.0], [50.0, 39.0]])
print(validator.naive_RMSE(result, ground_truth))

renderer.vectorfield_2D(directions)
renderer.trajectory_2D(result)
renderer.trajectory_2D(ground_truth)

renderer.show()
