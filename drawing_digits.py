# %%
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from mnistsbi import simulator

from mnistsbi.simulator import s_shape
from mnistsbi.simulator import strike3
from mnistsbi.simulator import bezier3_path
from mnistsbi.simulator import raster
from mnistsbi import digits_specs

# %%
x = np.linspace(0, 1, 100)
plt.plot(x, s_shape(x, power=1.5))


# %%
controls = np.asarray([
    [5., 5.],
    [25., 20.],
    [15., 10.],
    [20., 25.],
])

x = strike3(controls, power=1.3)
_, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x[:, 0], x[:, 1])
ax.set(
    xlim=(0, 28),
    ylim=(0, 28),
)

# %%
# digit 8 coordinates:
points = digits_specs.DIGITS_8A_PATH_COORDS

raw_controls_p = 0.25 * points[:-1] + 0.75 * points[1:]
raw_controls_n = 0.75 * points[:-1] + 0.25 * points[1:]

extra_p = points[1:] - raw_controls_p + points[1:]
extra_n = points[:-1] - raw_controls_n + points[:-1]

controls_p = (raw_controls_p[:-1] + extra_n[1:]) / 2
controls_n = (raw_controls_n[1:] + extra_p[:-1]) / 2

fig, ax = plt.subplots(figsize=(8, 8))
ax.set(xlim=(0, 27), ylim=(0, 27))
ax.plot(points[:, 0], points[:, 1], "-")
ax.scatter(controls_p[:, 0], controls_p[:, 1], color="green")
ax.scatter(controls_n[:, 0], controls_n[:, 1], color="red")
# plt.savefig("images/digit_8_cubic_bezier_control_points.png")


# %%
fig, ax = plt.subplots(figsize=(8, 8))
ax.set(xlim=(0, 27), ylim=(0, 27))
for segment_controls in bezier3_path(points):
    segment = strike3(segment_controls, power=1., n_steps=20)
    ax.plot(segment[:, 0], segment[:, 1], 'o')
# plt.savefig("images/digit_8_cubic_bezier_path_segments.png")


# %%
canvas = raster(points)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set(xlim=(0, 27), ylim=(0, 27))
ax.imshow(canvas, interpolation="nearest", cmap=plt.cm.Greys_r)
# plt.savefig("images/digit_8_rasterized.png")


# %%
def path_coords_to_strike_params(coords):
    """Convert euclidean coords to a sequence of polar coords

    The first record holds the euclidean coords.

    Subsequent records hold the radius and angle of the vector to go from one
    point to the next.

    This parametrization is more natural to introduce some random variations
    that could better represent variability of hand gestures.
    """
    strike_params = np.empty_like(coords)
    # euclidean coords of starting point
    strike_params[0, :] = coords[0]
    for i in range(coords.shape[0] - 1):
        a, b = coords[i], coords[i + 1]
        v = b - a
        radius = np.linalg.norm(v)
        angle = np.arctan2(v[1], v[0])
        strike_params[i + 1] = np.array([radius, angle])
    return strike_params


def strike_params_to_path_coords(strike_params):
    coords = np.empty_like(strike_params)
    # euclidean coords of starting point
    coords[0, :] = strike_params[0]
    for i in range(1, coords.shape[0]):
        radius, angle = strike_params[i]
        v = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
        ])
        coords[i] = coords[i - 1] + v
    return coords


# %%
strike_params = path_coords_to_strike_params(points)
points_reco = strike_params_to_path_coords(strike_params)
np.testing.assert_allclose(points_reco, points, rtol=1e-5)


# %%
def apply_noise(spec):
    strike_params = path_coords_to_strike_params(points)
    strike_params[0, :] += np.random.randn(2)
    scales = np.random.uniform(
        size=strike_params[1:, 0].shape,
        low=0.8, high=1.1,
    )
    strike_params[1:, 0] *= scales
    angle_noise = np.random.randn(len(points) - 1)
    angle_noise -= angle_noise.mean()
    strike_params[1:, 1] += 0.02 * np.pi * angle_noise
    strike_params[1:, 1] += 0.05 * np.pi * np.random.randn(1)
    points_perturbed = strike_params_to_path_coords(strike_params)
    points_perturbed += 14 - points_perturbed.mean(axis=0)
    return points_perturbed


points_perturbed = apply_noise(points)
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))
ax0.set(xlim=(0, 27), ylim=(0, 27))
ax0.imshow(raster(points), interpolation="nearest", cmap=plt.cm.Greys_r)

ax1.set(xlim=(0, 27), ylim=(0, 27))
ax1.imshow(raster(points_perturbed), interpolation="nearest",
           cmap=plt.cm.Greys_r)
# %%
points_perturbed = points.copy()
points_perturbed += 0.9 * np.random.normal(size=points.shape)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 6))
ax0.set(xlim=(0, 27), ylim=(0, 27))
ax0.imshow(raster(points), interpolation="nearest", cmap=plt.cm.Greys_r)

ax1.set(xlim=(0, 27), ylim=(0, 27))
ax1.imshow(raster(points_perturbed), interpolation="nearest",
           cmap=plt.cm.Greys_r)


# %%
from time import perf_counter


def simulate():
    return raster(apply_noise(digits_specs.DIGITS_8A_PATH_COORDS))


n_simulations = 500
tic = perf_counter()
for i in range(n_simulations):
    simulate()
toc = perf_counter()

print(f"{n_simulations / (toc - tic):.1f} simulations per second")


# %%
from joblib import Parallel, delayed

n_simulations = 1000
tic = perf_counter()
Parallel(n_jobs=4)(
    delayed(simulate)() for i in range(n_simulations))
toc = perf_counter()

print(f"{n_simulations / (toc - tic):.1f} simulations per second")
# %%
