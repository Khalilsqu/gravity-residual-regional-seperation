import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
import numpy as np
from scipy import interpolate
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def plot_scatter_locations(df, x_coord, y_coord, ba):
    fig = px.scatter(
        df,
        x=x_coord,
        y=y_coord,
        color=ba,
        width=1000,
        height=1000
    )
    fig.update_yaxes(
        tickformat=",.0f",
        range=[df[y_coord].min()-20, df[y_coord].max()+20]
    )
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        tickformat=",.0f",
        range=[df[x_coord].min()-50, df[x_coord].max()+50]
    )

    fig.update_layout(
        xaxis_title="Easting UTM",
        yaxis_title="Northing UTM",
    )
    selected_data = plotly_events(
        fig,
        click_event=False,
        hover_event=False,
        select_event=True,
        override_height=1000,
        override_width=1000,
    )

    return selected_data


def plot_profile_original(df, x_coord, y_coord, ba, val_inter):
    df = df.sort_values([y_coord, x_coord])
    distance = np.cumsum(
        np.sqrt(np.diff(df[x_coord])**2+np.diff(df[y_coord])**2)
    )

    distance = np.concatenate(([0], distance))

    fig1 = go.Figure(data=go.Scatter(x=distance, y=df[ba]))
    fig1.update_layout(
        xaxis_title="Distance",
        yaxis_title="Bouguer Anomaly",

    )

    if distance.size == df[ba].size:

        f = interpolate.interp1d(distance, df[ba], kind='cubic')

        distance_inter = np.linspace(0, distance[-1], val_inter)

        ynew = f(distance_inter)

        fig2 = make_subplots(rows=1, cols=2)

        fig2.add_trace(go.Scatter(x=distance_inter, y=ynew))

        fig2.update_xaxes(title_text="Distance", row=1, col=1)
        fig2.update_yaxes(
            title_text="Bouguer Anomaly Interpolated", row=1, col=1)

        yf = fft(ynew, distance_inter.size)
        freq = fftfreq(distance_inter.size, distance_inter[1]-distance_inter[0])[
            :distance_inter.size//2]  # get freq axis

        fig2.add_trace(go.Scatter(
            x=freq[1:distance_inter.size//2],
            y=2.0/distance_inter.size * np.abs(yf[1:distance_inter.size//2])
        ),
            row=1, col=2
        )

        fig2.update_xaxes(title_text="Frequency in Hz", row=1, col=2)
        fig2.update_yaxes(
            title_text="Amplitude", row=1, col=2)

        fig2.update_layout(showlegend=False)

        return fig1, (fig2,
                      distance_inter,
                      ynew,
                      min(freq[1:distance_inter.size//2]),
                      max(freq[1:distance_inter.size//2]),
                      freq.max()
                      )

    else:
        return fig1, None


def plot_high_pass_filter(distance, filtered, total):
    fig2 = make_subplots(rows=1, cols=2)

    residual = total - filtered

    fig2.add_trace(go.Scatter(x=distance, y=filtered), row=1, col=1)

    fig2.update_xaxes(title_text="Distance", row=1, col=1)
    fig2.update_yaxes(
        title_text="Residual bouguer anomaly", row=1, col=1)

    regional = total - filtered

    fig2.add_trace(go.Scatter(x=distance, y=regional), row=1, col=2)

    fig2.update_xaxes(title_text="Distance", row=1, col=2)
    fig2.update_yaxes(
        title_text="Regional bouguer anomaly", row=1, col=2)

    fig2.update_layout(showlegend=False)

    return fig2


def scatter_map(regional, x, y, ba):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Regional Field", "Residual Field")
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(color=regional,
                        colorscale='Viridis', size=14,)),
        row=1,
        col=1,
    )

    residual = ba - regional

    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(color=residual,
                        colorscale='Viridis', size=14,)), row=1, col=2,
    )

    fig.update_yaxes(
        title_text="Northing UTM",
        tickformat=",.0f", row=1, col=2
    )
    fig.update_xaxes(
        title_text="Easting UTM",
        scaleanchor="y",
        scaleratio=1,
        tickformat=",.0f", row=1, col=2
    )

    fig.update_yaxes(
        title_text="Northing UTM",
        tickformat=",.0f", row=1, col=1
    )
    fig.update_xaxes(
        title_text="Easting UTM",
        scaleanchor="y",
        scaleratio=1,
        tickformat=",.0f", row=1, col=1
    )

    fig.update_layout(showlegend=False, height=800, width=800,)

    method = "cubic"
    dir_x_no = 50
    dir_y_no = 50

    xi = np.linspace(min(x), max(x), dir_x_no)
    yi = np.linspace(min(y), max(y), dir_y_no)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolation

    zi_ba = griddata((x, y), ba, (xi, yi), method=method)
    zi_reg = griddata((x, y), regional, (xi, yi), method=method)
    zi_res = griddata((x, y), residual, (xi, yi), method=method)

    fig2, ax = plt.subplots(3, 1, figsize=(8, 15))

    # plot CBA contours
    im1 = ax[0].contourf(xi, yi, zi_ba, levels=50, cmap="jet")
    im2 = ax[1].contourf(xi, yi, zi_reg, levels=50, cmap="jet")
    im3 = ax[2].contourf(xi, yi, zi_res, levels=50, cmap="jet")
    ax[0].set_title('Complete Bouguer Anomaly')
    ax[1].set_title('Regional Field')
    ax[2].set_title('Residual Field')

    fig2.colorbar(im1, ax=ax[0])
    fig2.colorbar(im2, ax=ax[1])
    fig2.colorbar(im3, ax=ax[2])

    for axes in ax:
        axes.set_xlabel('x')
        axes.set_ylabel('y')
        axes.ticklabel_format(useOffset=False, style='plain')

    return fig, fig2
