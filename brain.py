import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import csv
from scipy import signal, stats
import pandas as pd
import altair as alt

# Function to smooth data using a triangular filter
def smooth_triangle(data, degree):
    triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))  # up then down
    smoothed = []

    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point) / np.sum(triangle))

    # Handle boundaries
    smoothed = [smoothed[0]] * int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])

    return smoothed

# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.header("Brain Wave Analysis")
    st.caption('Enhance your understanding of mental wellness through EEG data analysis.')

    uploaded_file = st.file_uploader("Upload a WAV File", type=["wav"])

    if uploaded_file is not None:
        fs, data = waves.read(uploaded_file)

        length_data = np.shape(data)
        length_new = length_data[0] * 0.05
        ld_int = int(length_new)

        data_new = signal.resample(data, ld_int)

        # Spectrogram
        _, _, t, _ = plt.specgram(data_new, NFFT=256, Fs=500, noverlap=250)
        t_max = max(t)

        st.subheader("Alpha Power Over Time")
        position_vector = []
        length_f = np.shape(t)
        l_row_f = length_f[0]
        for i in range(0, l_row_f):
            if t[i] >= 7 and t[i] <= 12:
                position_vector.append(i)

        # Check if position_vector is not empty before proceeding
        if position_vector:
            length_d = np.shape(data_new)
            l_col_d = length_d[1] if len(length_d) > 1 else 0
            AlphaRange = [np.mean(data_new[position_vector[0]:min(position_vector[-1] + 1, l_col_d), i]) for i in range(l_col_d)]

            alpha_df = pd.DataFrame({'Time': t, 'Alpha Power': smooth_triangle(AlphaRange, 100)})

            alpha_chart = alt.Chart(alpha_df).mark_line().encode(
                x=alt.X('Time', scale=alt.Scale(domain=(0, t_max))),
                y='Alpha Power'
            ).properties(
                width=600,
                height=300
            )

            st.altair_chart(alpha_chart)

            st.markdown("### Alpha Power Over Time Explanation:")
            st.write("This plot shows the variation in alpha power (8-10 Hz) over time after smoothing.")
            st.write("Alpha waves are commonly associated with relaxed states.")
        else:
            st.warning("No data in the specified frequency range (7 to 12 Hz). Please check the input file.")

        # Continue with the rest of your code...


        # Statistical Analysis
        tg = np.array([4.2552, 14.9426, 23.2801, 36.0951, 45.4738, 59.3751, 72.0337, 85.0831, max(t) + 1])
        eyesclosed = []
        eyesopen = []
        j = 0  # Initial variable to traverse tg
        l = 0  # Initial variable to loop through the "y" data
        for i in range(0, l_row_f):  # Fix: Change 'l_row_t' to 'l_row_f'
            if t[i] >= tg[j]:
                if j % 2 == 0:
                    eyesopen.append(np.mean(alpha_df['Alpha Power'][l:i]))  # Fix: Change 'y' to 'alpha_df['Alpha Power']'
                if j % 2 == 1:
                    eyesclosed.append(np.mean(alpha_df['Alpha Power'][l:i]))  # Fix: Change 'y' to 'alpha_df['Alpha Power']'
                l = i
                j = j + 1

        statistical_df = pd.DataFrame({'Category': ['Eyes Open', 'Eyes Closed'], 'Alpha Power': [np.mean(eyesopen), np.mean(eyesclosed)]})

        statistical_chart = alt.Chart(statistical_df).mark_bar().encode(
            x='Category',
            y='Alpha Power'
        ).properties(
            width=400,
            height=300
        )

        st.altair_chart(statistical_chart)

        st.markdown("### Statistical Analysis Explanation:")
        st.write("The bar chart visually compares the mean alpha power during eyes open and eyes closed states.")
        st.write("The t-test result indicates statistical significance in the difference.")

        st.write("Mean (Eyes Open):", np.mean(eyesopen))
        st.write("Mean (Eyes Closed):", np.mean(eyesclosed))
        st.write("Standard Deviation (Eyes Open):", np.std(eyesopen))
        st.write("Standard Deviation (Eyes Closed):", np.std(eyesclosed))

        result = stats.ttest_ind(eyesopen, eyesclosed, equal_var=False)
        st.write("T-Test Result:")
        st.write("t-Statistic:", result.statistic)
        st.write("p-Value:", result.pvalue)

if __name__ == "__main__":
    main()
