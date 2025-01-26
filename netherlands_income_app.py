import streamlit as st
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

INFO_STRING = """
This is all based on the data from https://www.cbs.nl/nl-nl/visualisaties/inkomensverdeling from 2022.

This data is the 'gestandaardiseerd inkomen' distribution, which is the net income standardised by household type.

The standardisation factor is sqrt(num_adults + (0.8 * num_kids)).

This app simply unscales the income based on household type to give an estimated percentile.
"""

##############################################
### Parsing code for the original csv data ###
##############################################
# def parse_income_bin(entry: str) -> int:
#     splitup = entry.split()
#     if len(splitup) == 3:
#         if splitup[0] == "minder":
#             return [-8, int(splitup[-1])]
#         else:
#             return [int(splitup[-1]), 102]
#     else:
#         return [int(splitup[1]), int(splitup[3])]
# df = pd.read_csv("data.csv", sep=";")
# bins = np.array([parse_income_bin(a) for a in df["gestandaardiseerd inkomen (x 1 000 euro)"]])
# bin_starts = bins[:, 0]
# counts = df["Alle huishoudens"].values
# cdf = np.cumsum(df["Alle huishoudens"].values)/df["Alle huishoudens"].sum()

#####################################################################################
#### Hardcoded data from https://www.cbs.nl/nl-nl/visualisaties/inkomensverdeling ###
#####################################################################################
# fmt: off
bin_starts = np.array([ -8,  -6,  -4,  -2,   0,   2,   4,   6,   8,  10,  12,  14,  16,
        18,  20,  22,  24,  26,  28,  30,  32,  34,  36,  38,  40,  42,
        44,  46,  48,  50,  52,  54,  56,  58,  60,  62,  64,  66,  68,
        70,  72,  74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,
        96,  98, 100])
counts = np.array([  4,   1,   2,  21,  38,  46,  52,  52,  57,  62,  85, 158, 326,
        443, 479, 520, 456, 434, 434, 429, 418, 404, 384, 354, 317, 280,
        246, 215, 185, 158, 134, 113,  95,  80,  68,  57,  48,  41,  35,
        30,  26,  22,  19,  17,  15,  13,  12,  11,  10,   9,   8,   7,
         6,   6,  95])
# fmt: on
cdf = np.cumsum(counts) / counts.sum()
norm_counts = counts / counts.sum()


def equivalence_factor(num_adults: int = 1, num_kids: int = 0) -> float:
    # See https://www.cbs.nl/nl-nl/achtergrond/2008/50/wat-is-mijn-besteedbaar-inkomen-
    return np.sqrt(num_adults + (0.8 * num_kids))


def main():
    st.title("NL Estimated Income Percentiles")

    # Input fields for Net Income, Adults, and Children
    net_income = st.number_input(
        "Annual Net Income (x€1000) [use https://thetax.nl]", value=29.63
    )
    adults = st.number_input("Number of Adults", value=1, step=1, min_value=1, max_value=4)
    children = st.number_input("Number of Children", value=0, step=1, min_value=0, max_value=4)

    # Button to trigger calculation and plot
    if st.button("Generate Plot"):
        # Replace this with your custom function that produces a matplotlib figure
        fig = process(net_income, adults, children)

        # Display the matplotlib figure
        st.pyplot(fig)

    with st.expander("How was this calculated?"):
        st.write(INFO_STRING)


def process(net_income, na, nk):
    eqf = equivalence_factor(na, nk)
    incomes = bin_starts * eqf
    incomes_k = incomes * 1000

    interp_fn = interp1d(incomes, cdf, kind="linear", fill_value="extrapolate")
    pctile = float(np.clip(interp_fn(net_income), 0.0, 1.0)) * 100
    bin_idx = np.searchsorted(incomes, net_income, side="right") - 1
    bar_width = (incomes[1] - incomes[0]) / 2
    bar_colours = ["blue" if i != bin_idx else "orange" for i in range(len(incomes))]

    fig, _ = plt.subplots(dpi=300)
    plt.bar(incomes, norm_counts, width=bar_width, color=bar_colours)

    bar_height = norm_counts[bin_idx]
    bar_x = incomes[bin_idx]

    text = f"€{net_income}k net is in the {pctile:.1f} percentile\nof incomes for household type\n{na} Adult(s) and {nk} children"
    plt.annotate(
        text,
        xy=(bar_x, bar_height),
        xytext=(max(incomes), max(norm_counts)),
        arrowprops=dict(facecolor="black", arrowstyle="->"),
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow"),
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
    )
    plt.xlabel(f"Net Income (x€1000)")
    plt.ylabel("P(Income)")
    return fig


if __name__ == "__main__":
    main()
