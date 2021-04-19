# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# %%
def s_shape(x, power=1.5, eps=1e-7):
    """Helper function to densify the midle of a cubic Bezier curve"""
    return (
        1 / (
            1 + ((x + eps) / (1 - x + eps)) ** -power
        )
    )


x = np.linspace(0, 1, 100)
plt.plot(x, s_shape(x, power=1.5))


# %%
def bezier3(controls, t):
    assert controls.shape[0] == 4
    t = t[:, None]
    return (
        t ** 3 * controls[0]
        + (1 - t) ** 3 * controls[1]
        + 3 * t ** 2 * (1 - t) * controls[2]
        + 3 * t * (1 - t) ** 2 * controls[3]
    )


def strike3(controls, power=1.5, n_steps=30):
    t = s_shape(np.linspace(0, 1, n_steps), power=power)
    return bezier3(controls, t)


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
points = np.asarray([
    [7, 22],
    [20, 10],
    [20, 5],
    [10, 5],
    [8, 12],
    [17, 21],
    [10, 23],
    [7.5, 22],
])

points = np.asarray([
    [14, 13],
    [8, 17],
    [12, 23],
    [19, 18],
    [10, 10],
    [8, 5],
    [13, 3],
    [19, 6],
    [14, 13],
])

points = np.asarray([
    [11, 19],
    [20, 20],
    [20, 20],
    [8, 4],
])
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
def bezier3_path(points, alpha=0.3):
    """Draw a smooth cubic Bezier path connecting points

    Control points are symmetric around each segment junction points
    on the path.
    """
    raw_controls_p = alpha * points[:-1] + (1 - alpha) * points[1:]
    raw_controls_n = (1 - alpha) * points[:-1] + alpha * points[1:]
    extra_p = points[1:] - raw_controls_p + points[1:]
    extra_n = points[:-1] - raw_controls_n + points[:-1]
    controls_p = (raw_controls_p[:-1] + extra_n[1:]) / 2
    controls_n = (raw_controls_n[1:] + extra_p[:-1]) / 2

    segments = []
    for i in range(points.shape[0] - 1):
        bezier_params = np.empty(shape=(4, 2), dtype=np.float64)
        bezier_params[0] = points[i]
        bezier_params[1] = points[i + 1]
        if i == 0:
            bezier_params[2] = points[i]
            bezier_params[3] = controls_p[i]
        elif i == points.shape[0] - 2:
            bezier_params[2] = controls_n[i - 1]
            bezier_params[3] = points[i + 1]
        else:
            bezier_params[2] = controls_n[i - 1]
            bezier_params[3] = controls_p[i]
        segments.append(bezier_params)
    return segments


fig, ax = plt.subplots(figsize=(8, 8))
ax.set(xlim=(0, 27), ylim=(0, 27))
for segment_controls in bezier3_path(points):
    segment = strike3(segment_controls, power=1., n_steps=20)
    ax.plot(segment[:, 0], segment[:, 1], 'o')
# plt.savefig("images/digit_8_cubic_bezier_path_segments.png")


# %%
def raster(points, grid_shape=(28, 28), width=.4, saturation=.8,
           n_steps_per_segment=25):
    curve = np.concatenate([
        strike3(c, power=1., n_steps=n_steps_per_segment)
        for c in bezier3_path(points)
    ])
    canvas = np.zeros(shape=grid_shape, dtype=np.float64)
    x = np.arange(grid_shape[0])
    y = np.arange(grid_shape[1])
    xx, yy = np.meshgrid(x, y)
    coords = np.array([xx, yy]).transpose(1, 2, 0)
    for i in range(curve.shape[0]):
        canvas += multivariate_normal.pdf(coords, curve[i], cov=width)
    return canvas.clip(min=0, max=saturation) / saturation


canvas = raster(points)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set(xlim=(0, 27), ylim=(0, 27))
ax.imshow(canvas, interpolation="nearest", cmap=plt.cm.Greys_r)
# plt.savefig("images/digit_8_rasterized.png")

# %%
