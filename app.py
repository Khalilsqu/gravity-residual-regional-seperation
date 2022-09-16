import streamlit as st
from scipy import signal

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from utils_funs import estimate_trend
from utils_funs import file_upload
import plots


st.set_page_config(
    "Gravity",
    page_icon="🌍",
    layout="wide",

)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
            content:'Made by Khalil Al Hooti'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.write(
    '<h1 style="color:blue;">Gravity residual seperation</h1>',
    unsafe_allow_html=True
)
st.write(
    '<h3 >This app seperates regional and residual gravity field from calculated bouguer gravity anomaly.</h3>',
    unsafe_allow_html=True
)

st.write(
    '<h2 style="color:green;">Read a table and choose the the four columns x-coord, y-coord, BA and station number</h2>',
    unsafe_allow_html=True
)
st.write(
    """<table>
  <tr>
    <td>x-coord</td>
    <td>y-coord</td>
    <td>Complete BA</td>
    <td>Station #</td>
  </tr>
  <tr>
    <td>:</td>
    <td>:</td>
    <td>:</td>
    <td>:</td>
  </tr>
</table>""",
    unsafe_allow_html=True
)


sheet_file = st.file_uploader(
    "",
    type=[
        'xlsx',
        'csv',
        'txt'
    ],
    accept_multiple_files=False,
    help="""Choose a sheet that contains columns of x, y, 
     complete bouguer anomaly and station number""",
)

if sheet_file:
    if "df" not in st.session_state:
        st.session_state['df'] = file_upload(sheet_file)
    if "df" in st.session_state:
        st.session_state['df'].dropna(how='all', axis=1, inplace=True)
        st.session_state['df'].dropna(how='all', axis=0, inplace=True)

        cols = st.columns(4)

        x_coord = cols[0].selectbox(
            "X-coordinate",
            st.session_state['df'].columns
        )

        y_coord = cols[1].selectbox(
            "Y-coordinate",
            st.session_state['df'].columns
        )

        ba = cols[2].selectbox(
            "BA",
            st.session_state['df'].columns
        )

        station = cols[3].selectbox(
            "Station",
            st.session_state['df'].columns
        )

        if len(set([x_coord, y_coord, ba, station])) != 4:
            st.error("cannot choose the same columns twice")
        else:
            st.write(
                '<h2 style="color:red;">Use the "Lasso Select" to select a profile</h2>',
                unsafe_allow_html=True
            )
            selected_data = plots.plot_scatter_locations(
                st.session_state['df'],
                x_coord,
                y_coord,
                ba,
            )
            x_coord_data = []
            y_coord_data = []

            for adata in selected_data:
                x_coord_data.append(adata['x'])
                y_coord_data.append(adata['y'])

            lasio_sel = st.session_state['df'][st.session_state['df'][x_coord].isin(
                x_coord_data) & st.session_state['df'][y_coord].isin(y_coord_data)]

            lasio_sel[ba] = lasio_sel.groupby(
                station)[ba].transform('mean')

            lasio_sel = lasio_sel.drop_duplicates(
                station).reset_index(drop=True)

            values_inter = st.number_input("Choose the number of interpolation points",
                                           min_value=20, max_value=300, value=100)

            fig1, second_data = plots.plot_profile_original(
                lasio_sel, x_coord, y_coord, ba, values_inter
            )

            st.write(
                '<h2 style="color:blue;">Original Profile</h2>',
                unsafe_allow_html=True
            )
            st.plotly_chart(fig1, use_container_width=True)

            if second_data is not None:
                fig2, distance_inter, ynew, minfreq, maxfreq, nyq = second_data
                st.write(
                    """<h2 style="color:blue;">Interpolated Profile and its
                     frequency spectrum</h2>""",
                    unsafe_allow_html=True
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.write(
                    """<h2 style="color:blue;">Residual and Regional
                     fields</h2>""",
                    unsafe_allow_html=True
                )

                cutoff_freq = st.number_input("Cutoff Frequency",
                                              min_value=0.000001,
                                              max_value=100.,
                                              value=(maxfreq-minfreq)/3,
                                              step=0.0001,
                                              )

                normal_cutoff = cutoff_freq / nyq

                b, a = signal.butter(
                    5, normal_cutoff, btype="high", analog=False)
                filtered = signal.filtfilt(b, a, ynew)

                fig3 = plots.plot_high_pass_filter(
                    distance_inter, filtered, ynew)

                st.plotly_chart(fig3, use_container_width=True)

                st.write(
                    """<h2 style="color:blue;">Residual and Regional fields of
                     all data as maps</h2>""",
                    unsafe_allow_html=True
                )

                degree_poly_fit = st.number_input(
                    "choose the degree of smoothing",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    help="""The larger this number the more dominant is the
                     regional field compared to residual field
                    """
                )

                regional = estimate_trend(
                    st.session_state['df'][x_coord],
                    st.session_state['df'][y_coord],
                    st.session_state['df'][ba],
                    degree=degree_poly_fit
                )
                fig4, fig5 = plots.scatter_map(
                    regional,
                    st.session_state['df'][x_coord].values,
                    st.session_state['df'][y_coord].values,
                    st.session_state['df'][ba].values,
                )

                st.plotly_chart(fig4, use_container_width=True)
                st.pyplot(fig5)
