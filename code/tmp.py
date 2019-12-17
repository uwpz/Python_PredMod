
'''
plot_distr(df, [metr[0], cate[0]], target_type = TARGET_TYPE,
           varimp = None, color = twocol, ylim = None, min_width = 0,
           nrow = 2, ncol = 2, w = 12, h = 6,
           pdf = None)  # maybe plot miss separately
'''
df = df; features = [metr[0], cate[0]]; target = "target"; target_type="REGR"; color=twocol; varimp=None; min_width=0
regplot = False; ylim = None; ncol=2; nrow=2; pdf=None; w=8; h=6



# Help variables
n_ppp = ncol * nrow  # plots per page


# Plot (loop over features)
for i, feature_act in enumerate(features):
    # i=0; feature_act=features[i]

    # Start new subplot on new page
    if i % n_ppp == 0:
        fig, ax = plt.subplots(nrow, ncol)
        fig.set_size_inches(w = w, h = h)
        i_ax = 0

    # Catch single plot case
    if n_ppp == 1:
        ax_act = ax
    else:
        ax_act = ax.flat[i_ax]

    # Metric feature
    if df[feature_act].dtype != "object":

        # Main Heatmap

        # Calc scale
        if ylim is not None:
            ax_act.set_ylim(ylim)
            tmp_scale = (ylim[1] - ylim[0]) / (np.max(df[target]) - np.min(df[target]))
        else:
            tmp_scale = 1

        # Calc colormap
        tmp_cmap = mcolors.LinearSegmentedColormap.from_list("gr_bl_yl_rd",
                                                             [(0.5, 0.5, 0.5, 0), "blue", "yellow",
                                                              "red"])
        # Hexbin plot
        ax_act.set_facecolor('0.98')
        p = ax_act.hexbin(df[feature_act], df[target],
                          gridsize = (int(50 * tmp_scale), 50),
                          cmap = tmp_cmap)
        plt.colorbar(p, ax = ax_act)
        if varimp is not None:
            ax_act.set_title(feature_act + " (VI: " + str(varimp[feature_act]) + ")")
        else:
            ax_act.set_title(feature_act)
        ax_act.set_ylabel(target)
        ax_act.set_xlabel(feature_act + " (NA: " +
                          str(df[feature_act].isnull().mean().round(3) * 100) +
                          "%)")
        ylim = ax_act.get_ylim()
        # ax_act.grid(False)
        ax_act.axhline(color = "grey")

        # Add lowess regression line?
        if regplot:
            sns.regplot(feature_act, target, df, lowess = True, scatter = False, color = "black", ax = ax_act)

        # # Inner Histogram
        # ax_act.set_ylim(ylim[0] - 0.4 * (ylim[1] - ylim[0]))
        inset_ax = ax_act.inset_axes([0, 0.07, 1, 0.2])
        inset_ax.set_axis_off()
        # #inset_ax.get_yaxis().set_visible(False)
        # #inset_ax.get_xaxis().set_visible(False)
        ax_act.get_shared_x_axes().join(ax_act, inset_ax)
        # i_bool = df[feature_act].notnull()
        # sns.distplot(df[feature_act].dropna(), bins = 20, color = "black", ax = inset_ax)
        # #inset_ax.set_xlabel("")
        #
        # # Inner-inner Boxplot
        # #inset_ax = ax_act.inset_axes([0, 0.01, 1, 0.05])
        # #inset_ax.set_axis_off()
        # inset_ax.get_yaxis().set_visible(False)
        # #inset_ax.get_xaxis().set_visible(False)
        # #inset_ax.get_shared_x_axes().join(ax_act, inset_ax)
        # #inset_ax.set_axis_off()
        # #sns.boxplot(x = df.loc[i_bool, feature_act], palette = ["grey"], ax = inset_ax)
        # # inset_ax.set_xlabel("")
        #
        # print(ax_act.__repr__)
        #
        # inset_ax.set_xlabel(feature_act + " (NA: " +
        #                     str(df[feature_act].isnull().mean().round(3) * 100) +
        #                     "%)")

    # Categorical feature
    else:
        # Prepare data (Same as for CLASS target)
        df_plot = pd.DataFrame({"h": df.groupby(feature_act)[target].mean(),
                                "w": df.groupby(feature_act).size()}).reset_index()
        df_plot["pct"] = 100 * df_plot["w"] / len(df)
        df_plot["w"] = 0.9 * df_plot["w"] / max(df_plot["w"])
        df_plot[feature_act + "_new"] = (df_plot[feature_act] + " (" +
                                         (df_plot["pct"]).round(1).astype(str) + "%)")
        df_plot["new_w"] = np.where(df_plot["w"].values < min_width, min_width, df_plot["w"])

        # Main grouped boxplot
        if ylim is not None:
            ax_act.set_xlim(ylim)
        bp = df[[feature_act, target]].boxplot(target, feature_act, vert = False,
                                               widths = df_plot.w.values,
                                               showmeans = True,
                                               meanprops = dict(marker = "x",
                                                                markeredgecolor = "black"),
                                               flierprops = dict(marker = "."),
                                               return_type = 'dict',
                                               ax = ax_act)
        [[item.set_color('black') for item in bp[target][key]] for key in bp[target].keys()]
        fig.suptitle("")
        ax_act.set_xlabel(target)
        ax_act.set_yticklabels(df_plot[feature_act + "_new"].values)
        if varimp is not None:
            ax_act.set_title(feature_act + " (VI: " + str(varimp[feature_act]) + ")")
        else:
            ax_act.set_title(feature_act)
        ax_act.axvline(np.mean(df[target]), ls = "dotted", color = "black")

        # Inner barplot
        xlim = ax_act.get_xlim()
        ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
        inset_ax = ax_act.inset_axes([0, 0, 0.2, 1])
        inset_ax.get_shared_y_axes().join(ax_act, inset_ax)
        inset_ax.set_axis_off()
        if ylim is not None:
            ax_act.axvline(ylim[0], color = "black")

        print(ax_act.__repr__)

        # df_plot.plot.barh(y = "w", x = feature_act, sharex = False, sharey=False, subplots=False,
        #                  color = "lightgrey", ax = inset_ax, edgecolor = "black", linewidth = 1,
        #                  legend = False)


        inset_ax.barh(df_plot.index.values+1, df_plot.w, color="lightgrey", edgecolor="black",
                      linewidth=1)

    i_ax += 1

    fig.tight_layout()


