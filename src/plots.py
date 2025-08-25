import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib import rc_context, colors, lines, patches, ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
import shapely

def plot_sda_probability_by_distance(b_vals, h_vals, fixed_b, fixed_h, colormap=None):
    custom = {
        "text.usetex": False,
        "pdf.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "grid.linewidth": 0.5,
        "grid.color": "black",
        "patch.linewidth": 0.5,
        "patch.edgecolor": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5
    }

    distances = np.linspace(0, 1, 1000)
    colors_b = [colormap(i / len(b_vals)) for i in range(len(b_vals))]
    colors_h = [colormap(i / len(h_vals)) for i in range(len(h_vals))]

    with rc_context(rc=custom):
        fig, axes = plt.subplots(ncols=2, figsize=(6.5, 3), sharey=True, constrained_layout=True)

        for b, color in zip(b_vals, colors_b):
            prob = 1 / (1 + (distances / b) ** fixed_h)
            axes[0].plot(distances, prob, label=f"$b={b}$", color=color, linewidth=1)
        
        for h, color in zip(h_vals, colors_h):
            prob = 1 / (1 + (distances / fixed_b) ** h)
            axes[1].plot(distances, prob, label=f"$h={h}$", color=color, linewidth=1)

        legend = axes[0].legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=1, borderaxespad=0, frameon=True, fancybox=False)
        legend.set_title(f"Fixed $h={fixed_h}$", prop={"size": 8.5})
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.5)

        legend = axes[1].legend(loc="upper right", bbox_to_anchor=(1, 1), ncol=1, borderaxespad=0, frameon=True, fancybox=False)
        legend.set_title(f"Fixed $b={fixed_b}$", prop={"size": 8.5})
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(0.5)

        axes[0].set_title(f"\nEffect of Characteristic Length Scale ($b$)\n", fontsize=8.5)
        axes[1].set_title(f"\nEffect of Homophily ($h$)\n", fontsize=8.5)

        axes[0].set_ylabel("\nSDA Connection Probability\n")
        axes[1].yaxis.set_label_position("right")
        axes[1].set_ylabel("\n \n", fontsize=8.5)
        axes[1].tick_params(right=False)

        axes[0].set_xlabel("\nDistance\n")
        axes[1].set_xlabel("\nDistance\n")

        for ax in axes:
            divider = lines.Line2D([0, 1], [1.05, 1.05], transform=ax.transAxes,
                                   color="black", linewidth=0.5, clip_on=False)
            ax.add_artist(divider)

        return fig

def plot_synthetic_locations(locations, color=None):
    custom = {
        "text.usetex": False,
        "pdf.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "grid.linewidth": 0.5,
        "grid.color": "black",
        "patch.linewidth": 0.5,
        "patch.edgecolor": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5
    }

    x, y = locations[:, 0], locations[:, 1]

    with rc_context(rc=custom):
        fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
        ax.scatter(x, y, s=10, edgecolors="black", color=color, alpha=0.75, linewidth=0.5)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_visible(False)

        return fig

def plot_geographic_locations(dataframe, locations, color=None):
    custom = {
        "text.usetex": False,
        "pdf.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "grid.linewidth": 0.5,
        "grid.color": "black",
        "patch.linewidth": 0.5,
        "patch.edgecolor": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5
    }

    df = dataframe.copy()
    points = [shapely.Point(x, y) for x, y in locations]
    locations_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:3435")
    locations_gdf = locations_gdf.to_crs(df.crs)

    with rc_context(rc=custom):
        fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
        df.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=0.5)
        locations_gdf.plot(ax=ax, markersize=0.25, color=color, alpha=0.75, linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_visible(False)

        return fig

def plot_joint_degree_distributions(dataframe, params, labels, binsizes, binby, refine=None, log=True, density=False, palette=None):
    custom = {
        "text.usetex": False,
        "pdf.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "grid.linewidth": 0.5,
        "grid.color": "black",
        "patch.linewidth": 0.5,
        "patch.edgecolor": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5
    }

    df = dataframe.copy()
    for key, value in (refine or {}).items():
        df = df[df[key] == value]

    model_ids = df["model_id"].unique()
    total_rows = sum(len(cfg["rows"]) for cfg in params)
    n_cols = len(model_ids)

    with rc_context(rc=custom):
        fig, axes = plt.subplots(total_rows, n_cols, figsize=(6.5, 2 * total_rows),
                                 sharex=False, sharey=True, constrained_layout=True)

        if total_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif total_rows == 1:
            axes = axes.reshape(1, n_cols)
        elif n_cols == 1:
            axes = axes.reshape(total_rows, 1)

        row_offset = 0
        for block_idx, block in enumerate(params):
            row_keys = block["rows"]
            row_labels = labels[block_idx]["rows"]
            col_label = labels[block_idx]["columns"]
            bin_axis = binby[block_idx]
            binsize = binsizes[block_idx]

            max_deg = df[bin_axis].max()
            bin_edges = np.arange(0, max_deg + binsize, binsize)

            for col_idx, model_id in enumerate(model_ids):
                model_df = df[df["model_id"] == model_id]

                for local_row, (key, ylabel) in enumerate(zip(row_keys, row_labels)):
                    global_row = row_offset + local_row
                    ax = axes[global_row, col_idx]

                    repeat = params[block_idx].get("repeat", False)
                    if repeat:
                        degrees = model_df[bin_axis].values
                        counts = model_df[key].values.astype(int)
                        expanded = np.repeat(degrees, counts)
                        ax.hist(expanded, bins=bin_edges, density=density,
                                color=palette.get(model_id, None), edgecolor="black")
                    else:
                        bin_groups = pd.cut(model_df[bin_axis], bins=bin_edges, right=False, include_lowest=True)
                        grouped = model_df.groupby(bin_groups, dropna=False, observed=False)[key].sum()
                        mids = [interval.mid if isinstance(interval, pd.Interval) else np.nan for interval in grouped.index]
                        ax.bar(mids, grouped.values, width=binsize,
                               color=palette.get(model_id, None), edgecolor="black")
                    
                    if log:
                        ax.set_yscale("log")
                    ax.tick_params(axis="both", labelsize=8.5)

                    if global_row == 0:
                        ax.set_title(f"\n{model_id}\n", fontsize=8.5)
                    elif local_row == 0:
                        ax.set_title("\n")

                    if local_row == len(row_keys) - 1 and col_idx == 1:
                        ax.set_xlabel(f"\n{col_label}\n", fontsize=8.5)
                    else:
                        ax.set_xlabel("")

                    if col_idx == 0:
                        ax.set_ylabel(f"\n{ylabel}\n", fontsize=8.5)
                    elif col_idx == n_cols - 1:
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel("\n \n", fontsize=8.5)
                        ax.tick_params(right=False)
                    else:
                        ax.set_ylabel("")

            for j in range(n_cols):
                ax = axes[row_offset, j]
                line = lines.Line2D([0, 1], [1.05, 1.05], transform=ax.transAxes,
                                    color="black", linewidth=0.5, clip_on=False)
                ax.add_artist(line)
                
            row_offset += len(row_keys)

        return fig

def plot_global_metrics(dataframe, params, labels, refine=None, aggfunc="mean", colormap=None):
    custom = {
        "text.usetex": False,
        "pdf.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "grid.linewidth": 0.5,
        "grid.color": "black",
        "patch.linewidth": 0.5,
        "patch.edgecolor": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5
    }

    model_ids = dataframe["model_id"].unique()
    n_rows = len(params)
    n_cols = len(model_ids)

    df = dataframe.copy()
    for key, value in (refine or {}).items():
        df = df[df[key] == value]

    with rc_context(rc=custom):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5, 3 * n_rows),
                                 sharex=False, sharey=False, constrained_layout=True)

        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        for i, (param_cfg, label_cfg) in enumerate(zip(params, labels)):
            row_key = param_cfg["rows"]
            col_key = param_cfg["columns"]
            val_key = param_cfg["value"]
            row_label = label_cfg["rows"]
            col_label = label_cfg["columns"]
            val_label = label_cfg["value"]

            pivot_vals = []
            for model_id in model_ids:
                sub = df[df["model_id"] == model_id]
                pivot = sub.pivot_table(index=row_key, columns=col_key,
                                        values=val_key, aggfunc=aggfunc)
                pivot_vals.append(pivot.values)

            all_values = np.concatenate(pivot_vals)
            norm = colors.TwoSlopeNorm(vcenter=np.nanmean(all_values),
                                       vmin=0, vmax=np.nanmax(all_values))

            for j, model_id in enumerate(model_ids):
                ax = axes[i, j]
                sub = df[df["model_id"] == model_id]
                pivot = sub.pivot_table(index=row_key, columns=col_key,
                                        values=val_key, aggfunc=aggfunc).sort_index()

                sns.heatmap(pivot, ax=ax, norm=norm, cmap=colormap,
                            linewidths=0.5, vmin=norm.vmin, vmax=norm.vmax)
                
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.5)
                    spine.set_edgecolor("black")

                ax.invert_yaxis()
                ax.tick_params(axis="both", labelsize=8.5)
                [tick.label1.set_rotation(0) for tick in ax.yaxis.get_major_ticks()]

                if j > 0:
                    ax.set_yticklabels([])
                if col_key == "model_id":
                    ax.set_xticks([])
                    ax.set_xticklabels([])

                ax.set_title(f"\n{model_id}\n", fontsize=8.5) if i == 0 else ax.set_title("\n")
                ax.set_xlabel(f"\n{col_label}\n", fontsize=8.5) if j == 1 else ax.set_xlabel("")
                ax.set_ylabel(f"\n{row_label}\n", fontsize=8.5) if j == 0 else ax.set_ylabel("")

                if j == 2:
                    cbar = ax.collections[-1].colorbar
                    cbar.set_label(f"\n{val_label}\n", fontsize=8.5)
                    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
                    for spine in cbar.ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(0.5)
                        spine.set_edgecolor("black")
                else:
                    ax.collections[-1].colorbar.ax.set_visible(False)
                
        divider_rows = [i for i in range(n_rows)]
        for i in divider_rows:
            for j in range(n_cols):
                ax = axes[i, j]
                line = lines.Line2D([0, 1], [1.05, 1.05], transform=ax.transAxes,
                                    color="black", linewidth=0.5, clip_on=False)
                ax.add_artist(line)

        return fig

def plot_local_metrics(dataframe, params, labels, mode, threshold=0.90, degs=("deg1", "deg2"), colormap=None):
    custom = {
        "text.usetex": False,
        "pdf.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.linewidth": 0.5,
        "axes.edgecolor": "black",
        "grid.linewidth": 0.5,
        "grid.color": "black",
        "patch.linewidth": 0.5,
        "patch.edgecolor": "black",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5
    }

    df = dataframe.copy()
    model_ids = df["model_id"].unique()
    n_rows = len(params)
    n_cols = len(model_ids)

    all_values = []
    visible_x = []
    visible_y = []

    for param in params:
        unique_id = param["unique_id"]
        for model_id in model_ids:
            sub = df[(df["unique_id"] == unique_id) & (df["model_id"] == model_id)]
            valid = sub[(~sub[mode].isna()) & (sub["joint_deg_number_of_overlap_edges_post"] > 0)]
            values = valid[mode].dropna()
            if not values.empty:
                all_values.append(values)
                visible_x.append(valid[degs[0]])
                visible_y.append(valid[degs[1]])

    all_values = np.concatenate(all_values)
    norm = colors.TwoSlopeNorm(vcenter=np.nanmean(all_values),
                                vmin=0, vmax=np.nanmax(all_values))
    
    x_vals = pd.concat(visible_x)
    y_vals = pd.concat(visible_y)
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    buffer = 10

    with rc_context(rc=custom):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5, 2 * n_rows),
                                 sharex=True, sharey=True, constrained_layout=True)

        if n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        for i, param in enumerate(params):
            unique_id = param["unique_id"]
            N, alpha, b, h = param["legend"]

            for j, model_id in enumerate(model_ids):
                ax = axes[i, j]
                sub = df[(df["unique_id"] == unique_id) & (df["model_id"] == model_id)]

                valid = sub[(~sub[mode].isna()) & (sub["joint_deg_number_of_nodes_post"] > 0)]

                x = valid[degs[0]].values
                y = valid[degs[1]].values
                c = valid[mode].values

                sorted_idx = np.argsort(c)
                ax.scatter(x[sorted_idx], y[sorted_idx], c=c[sorted_idx],
                           cmap=colormap, s=5, edgecolors="none", norm=norm)

                ax.set_xlim(-buffer, x_max + buffer)
                ax.set_ylim(-buffer, y_max + buffer)
                ax.tick_params(axis="both", labelsize=8.5)
                [tick.label1.set_rotation(0) for tick in ax.yaxis.get_major_ticks()]

                if i == 0:
                    ax.set_title(f"\n{model_id}\n", fontsize=8.5)
                    line = lines.Line2D([0, 1], [1.05, 1.05], transform=ax.transAxes,
                                        color="black", linewidth=0.5, clip_on=False)
                    ax.add_artist(line)

                ax.set_xlabel(f"\n{labels['columns']}\n", fontsize=8.5) if i == n_rows - 1 and j == 1 else ax.set_xlabel("")
                ax.set_ylabel(f"\n{labels['rows']}\n", fontsize=8.5) if i == 1 and j == 0 else ax.set_ylabel("")

                if j == n_cols - 1:
                    param_text = f"$N={N}$\n$\\alpha={alpha}$\n$b={b}$\n$h={h}$"
                    handle = lines.Line2D([], [], marker="", linestyle="none")
                    legend = ax.legend(
                        [handle], [param_text],
                        loc="upper left", bbox_to_anchor=(1.05, 1),
                        frameon=True, fancybox=False,
                        borderaxespad=0, handlelength=0, handletextpad=0.0,
                        fontsize=8.5
                    )
                    frame = legend.get_frame()
                    frame.set_edgecolor("black")
                    frame.set_linewidth(0.5)

                focus = valid["joint_deg_number_of_nodes_post"].values
                sorted = np.argsort(focus)[::-1]
                cum = np.cumsum(focus[sorted])
                total = focus.sum()
                cutoff_idx = np.where(cum <= threshold * total)[0]

                if len(cutoff_idx) > 0:
                    top_points = valid.iloc[sorted[cutoff_idx]]
                    rmin, rmax = top_points[degs[1]].min(), top_points[degs[1]].max()
                    cmin, cmax = top_points[degs[0]].min(), top_points[degs[0]].max()

                    box = patches.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin,
                                            linewidth=0.5, edgecolor="black",
                                            facecolor="none", linestyle="solid")
                    ax.add_patch(box)

                    inset_ax = inset_axes(ax, width="100%", height="100%",
                                          bbox_to_anchor=(0.05, 0.7, 0.45, 0.3),
                                          bbox_transform=ax.transAxes,
                                          loc="upper left", borderpad=0)

                    dense = valid[(valid[degs[1]] >= rmin) & (valid[degs[1]] <= rmax) &
                                  (valid[degs[0]] >= cmin) & (valid[degs[0]] <= cmax)].copy()

                    pivot = dense.pivot_table(index=degs[1], columns=degs[0], values=mode)
                    pivot.index = range(pivot.shape[0])
                    pivot.columns = range(pivot.shape[1])
                    mask = pivot.isna()

                    sns.heatmap(pivot, ax=inset_ax, mask=mask, cmap=colormap,
                                norm=norm, cbar=False, xticklabels=False,
                                yticklabels=False, linewidths=0.5)

                    inset_ax.set_xticks([0, pivot.shape[1] - 1])
                    inset_ax.set_xticklabels([f"{pivot.columns[0]}", f"{pivot.columns[-1]}"])

                    inset_ax.set_yticks([0, pivot.shape[0] - 1])
                    inset_ax.set_yticklabels([f"{pivot.index[0]}", f"{pivot.index[-1]}"])
                    inset_ax.tick_params(axis='both', labelsize=6.5)
                    inset_ax.yaxis.tick_right()
                    inset_ax.invert_yaxis()
                    inset_ax.set_facecolor("white")
                    for spine in inset_ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor("black")
                        spine.set_linewidth(0.5)

        cbar = fig.colorbar(ax.collections[0], ax=axes[:, -1],
                            location="right", shrink=1, aspect=60)
        cbar.set_label(f"\n{labels['value']}\n", fontsize=8.5)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        for spine in cbar.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("black")

        return fig