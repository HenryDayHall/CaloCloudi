import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


nice_hex = [
    ["#00E5E3", "#8DD7BF", "#FF96C5", "#FF5768", "#FFBF65"],
    ["#FC6238", "#FFD872", "#F2D4CC", "#E77577", "#6C88C4"],
    ["#C05780", "#FF828B", "#E7C582", "#00B0BA", "#0065A2"],
    ["#00CDAC", "#FF6F68", "#FFDACC", "#FF60A8", "#CFF800"],
    ["#FF5C77", "#4DD091", "#FFEC59", "#FFA23A", "#74737A"],
]


def center_arrow(ax, dx, dy, frac=0.1, **arrowprops):
    bb = ax.get_window_extent()          # axes size in pixels
    r = bb.width / bb.height
    ux, uy = np.array([dx, dy]) / np.hypot(dx, dy)
    tip = (0.5 + frac * ux, 0.5 + frac * uy * r)
    ax.annotate("", xy=tip, xytext=(0.5, 0.5),
                xycoords=ax.transAxes, textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>",
                                color="k", **arrowprops))


def plot_line_with_devation(
    ax,
    colour,
    xs,
    ys,
    ys_up_displacement,
    ys_down_displacement=None,
    clip_to_zero=False,
    **line_kwargs,
):
    """
    Plot a line with a shaded region around it.
    The shaded region usually represents the standard deviation of the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    colour : str
        matplotlib colour to use for the line and shaded region.
    xs : array-like, 1D
        The x values to plot.
    ys : array-like, 1D
        The y values for the central line.
    ys_up_displacement : array-like, 1D
        The y values for the upper edge of the shaded region.
    ys_down_displacement : array-like, 1D, optional
        The y values for the lower edge of the shaded region.
        If not given, the same distance as for the upper edge is used.
    clip_to_zero : bool, optional
        If True, no line goes below 0.
    **line_kwargs
        Additional keyword arguments to pass to the `ax.plot` function
        of the central line.

    """
    if ys_down_displacement is None:
        ys_down_displacement = ys_up_displacement
    ys_low = ys - ys_down_displacement
    ys_high = ys + ys_up_displacement
    if clip_to_zero:
        ys = np.maximum(ys, 0)
        ys_low = np.maximum(ys_low, 0)
        ys_high = np.maximum(ys_high, 0)
    ax.plot(xs, ys, color=colour, **line_kwargs)
    ax.fill_between(xs, ys_low, ys_high, color=colour, alpha=0.2)


def plot_hist_with_devation(
    ax,
    colour,
    bins,
    counts,
    errors_up,
    errors_down=None,
    clip_to_zero=False,
    **hist_kwargs,
):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    reduced_args = hist_kwargs.copy()
    for key in ["color", "histtype", "weights"]:
        if key in reduced_args:
            del reduced_args[key]
    ax.hist(
        bin_centers,
        bins=bins,
        color=colour,
        weights=counts,
        histtype="step",
        **reduced_args,
    )

    if errors_down is None:
        errors_down = errors_up

    lower = counts - errors_down
    upper = counts + errors_up

    lower = np.repeat(lower, 2)
    upper = np.repeat(upper, 2)

    if clip_to_zero:
        lower = np.maximum(lower, 0)
        upper = np.maximum(upper, 0)

    bin_corners = np.repeat(bins, 2)[1:-1]

    ax.fill_between(bin_corners, lower, upper, color=colour, alpha=0.2)


def heatmap(
    arr,
    x_spec,
    y_spec,
    x_label=None,
    y_label=None,
    cbar_label=None,
    title=None,
    ax=None,
    **pcolour_kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    n_x_points, n_y_points = arr.shape
    if len(x_spec) == 2:
        # they are limits, transform to even range
        x_spec = np.linspace(*x_spec, n_x_points + 1)
    if len(y_spec) == 2:
        y_spec = np.linspace(*y_spec, n_y_points + 1)
    cmesh = ax.pcolormesh(x_spec, y_spec, arr.T, **pcolour_kwargs)
    cbar = fig.colorbar(cmesh)
    cbar.set_label(cbar_label)
    return fig, ax, cbar, cmesh


def heatmap_tile(arr, z, X_meshgrid, Y_meshgrid, ax, cmap=plt.cm.cool, alpha=0.5):
    """
    Plot a heatmap as a flat slice in a 3D plot

    Parameters
    ----------
    arr : 2D array
        The data to plot.
    z : float
        The z high to put it at
    Xmin : float
        The minimum x value.
    Ymin : float
        The minimum y value.
    Xstep_size : float
        The step size in x.
    Ystep_size : float
        The step size in y.
    cmap : matplotlib.colors.Colormap, optional
        The colormap to use.
    alpha : float, optional
        The alpha value for the heatmap.

    """

    Z_meshgrid = np.full_like(X_meshgrid, z)
    face_colors = cmap(arr)
    face_colors[:, :, -1] = alpha
    ax.plot_surface(
        X_meshgrid,
        Y_meshgrid,
        Z_meshgrid,
        facecolors=face_colors,
        linewidth=0,
    )


def heatmap_stack(
    arrs,
    zs,
    Xmin=0,
    Xmax=None,
    Ymin=0,
    Ymax=None,
    ax=None,
    cmap=plt.cm.cool,
    alpha=0.5,
):
    """
    Plot a series of heatmaps as flat slices in a 3D plot

    Parameters
    ----------
    arrs : 3D array
        The data to plot in a grid of (x, y).
    zs : float
        The z high to put each slice at
    Xmin : float, optional
        The minimum x value.
        Default is 0.
    Xmax : float, optional
        The maximum x value.
        Default is length of the second dimension of arrs.
    Ymin : float, optional
        The minimum y value.
        Default is 0.
    Ymax : float, optional
        The maximum y value.
        Default is length of the third dimension of arrs.
    cmap : matplotlib.colors.Colormap, optional
        The colormap to use.
    alpha : float, optional
        The alpha value for the heatmap.

    """
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)

    if Xmax is None:
        Xmax = arrs.shape[1]
    if Ymax is None:
        Ymax = arrs.shape[2]

    X_tile_centers = np.linspace(Xmin, Xmax, arrs.shape[1])
    Y_tile_centers = np.linspace(Ymin, Ymax, arrs.shape[2])
    X_meshgrid, Y_meshgrid = np.meshgrid(X_tile_centers, Y_tile_centers, indexing="ij")

    for z, arr, alph in zip(zs, arrs, alpha):
        heatmap_tile(arr, z, X_meshgrid, Y_meshgrid, ax, cmap, alph)

    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    z_padding = (max(zs) - min(zs)) * 0.1
    ax.set_zlim(min(zs) - z_padding, max(zs) + z_padding)
    ax.get_figure().canvas.draw()


def plot_event(
    event_n,
    cond_features,
    event,
    energy_scale=1,
    xy_scale=1,
    xlim=(-150, 150),
    ylim=(-150, 150),
    ax=None,
):
    """
    Plot a event in 3D

    Parameters
    ----------
    event_n : int
        The event number, for the title.
    cond_features : float
        The conditioning features of the event, also for the title.
    event : np.array (n_points, 4)
        The event to plot. The columns are (x, y, z, e).
    energy_scale : float, optional
        Scale factor to apply to energy before plotting.
        Default is 1.
    xz_scale : float, optional
        Scale factor to apply to x and z before plotting.
        Default is 1.
    xlim : tuple, optional
        The limits for the x axis.
        Default is (-150, 150).
    ylim : tuple, optional
        The limits for the y axis.
        Default is (-150, 150).
    ax : matplotlib.axes.Axes, optional
        The axes to plot on.
        If not given, a new figure is created.
        Default is None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The 3D axes the event was plotted on.

    """
    energy = event[:, 3]
    mask = energy > 0
    n_points = sum(mask)
    energy = energy[mask] * energy_scale
    xs = event[mask, 0] * xy_scale
    ys = event[mask, 1] * xy_scale
    zs = event[mask, 2]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, c=energy, s=energy, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # write a title
    observed_energy = energy.sum()
    formated_conditioning = ", ".join(f"{f:.2f}" for f in cond_features)
    ax.set_title(
        f"Evt: {event_n}, $n_{{pts}}$: {n_points}, conditioning: "
        + formated_conditioning
        + f"$E_{{vis}}$: {observed_energy:.2f}"
    )
    return ax


def blank_axes(ax):
    """
    Remove the ticks and labels, and lines from the axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to blank.

    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.set_frame_on(False)


def project_if_needed(element, element_type, sequence_len):
    is_elem = isinstance(element, element_type)
    if not is_elem:
        assert (
            len(element) == sequence_len
        ), f"Expected length {sequence_len}, but see {len(element)}; {element}"
        return element
    return [element] * sequence_len


class RatioPlots:
    def __init__(
        self,
        x_labels,
        truth_bin_heights,
        binnings,
        truth_label="g4",
        truth_color=(0.5, 0.5, 0.5, 0.5),
        max_cols=3,
        logx=False,
        logy=False,
    ):
        # set up plot axes
        self.n_features = len(x_labels)
        self.n_cols = min(self.n_features, max_cols)
        self.n_rows = int(np.ceil(self.n_features / self.n_cols))
        height_ratios = [3, 1] * self.n_rows
        self.fig, self.axes = plt.subplots(
            2 * self.n_rows,
            self.n_cols,
            figsize=(20, 5 * self.n_rows),
            gridspec_kw={"height_ratios": height_ratios},
        )
        if self.n_cols == 1:
            self.axes = self.axes[:, None]
        # ylabels to the left
        for row in range(self.n_rows):
            ax = self.axes[row * 2, 0]
            ax.set_ylabel("Counts")

        # store bins and truth counts
        self.bins = binnings
        self.truth_counts = truth_bin_heights
        self.bin_centers = [0.5 * (b[1:] + b[:-1]) for b in self.bins]

        # plot truth hists
        truth_hist_kwargs = dict(
            label=truth_label, histtype="stepfilled", color=truth_color
        )
        for i, label in enumerate(x_labels):
            row = int(i / self.n_cols)
            col = i - (row * self.n_cols)
            main_ax = self.axes[row * 2, col]
            main_ax.set_xlabel(label)
            main_ax.hist(
                self.bin_centers[i],
                bins=self.bins[i],
                weights=self.truth_counts[i],
                **truth_hist_kwargs,
            )
        self.logx = project_if_needed(logx, bool, self.n_features)
        self.logy = project_if_needed(logy, bool, self.n_features)

        # prep for plotting ratios
        self.ratio_min_maxes = [[1.0, 1.0] for _ in range(self.n_features)]
        for i in range(self.n_features):
            row = int(i / self.n_cols)
            col = i - (row * self.n_cols)
            ratio_ax = self.axes[row * 2 + 1, col]
            ratio_ax.hlines(1, self.bins[i][0], self.bins[i][-1], color=truth_color)

    def add_comparison(self, bin_heights, label, colour):
        model_hist_kwargs = dict(label=label, histtype="step", color=colour)
        model_ratio_kwargs = dict(label=label, c=colour)
        for i in range(self.n_features):
            row = int(i / self.n_cols)
            col = i - (row * self.n_cols)
            main_ax = self.axes[row * 2, col]
            main_ax.hist(
                self.bin_centers[i],
                bins=self.bins[i],
                weights=bin_heights[i],
                **model_hist_kwargs,
            )
            ratio = bin_heights[i] / self.truth_counts[i]
            ratio_ax = self.axes[row * 2 + 1, col]
            ratio_ax.plot(self.bin_centers[i], ratio, **model_ratio_kwargs)
            self.ratio_min_maxes[i][0] = min(
                self.ratio_min_maxes[i][0], np.nanmin(ratio)
            )
            self.ratio_min_maxes[i][1] = max(
                self.ratio_min_maxes[i][1], np.nanmax(ratio)
            )

    def finalise(self):
        self.axes[-2, -1].legend()
        for i, (y_min, y_max) in enumerate(self.ratio_min_maxes):
            clipped_min = max(y_min - 0.1, 0.0)
            clipped_max = min(y_max + 0.1, 2.0)
            clipped_min = 0.6
            clipped_max = 1.4
            row = int(i / self.n_cols)
            col = i - (row * self.n_cols)
            ratio_ax = self.axes[row * 2 + 1, col]
            ratio_ax.set_ylim(clipped_min, clipped_max)
            ratio_ax.set_xlim(self.bins[i][0], self.bins[i][-1])
            main_ax = self.axes[row * 2, col]
            main_ax.set_xlim(self.bins[i][0], self.bins[i][-1])
            if self.logx[i]:
                main_ax.set_xscale("log")
                ratio_ax.set_xscale("log")
            if self.logy[i]:
                main_ax.set_yscale("log")
        self.fig.tight_layout()
