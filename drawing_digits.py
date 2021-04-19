# %%
import numpy as np
import matplotlib.pyplot as plt

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
points = digits_specs.DIGITS_8A_PATH

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
