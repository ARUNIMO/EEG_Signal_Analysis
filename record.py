
import streamlit as st
import numpy as np
import scipy.io.wavfile as waves
from scipy import signal, stats
import plotly.graph_objects as go

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
        fig = go.Figure()

        f, t, Sxx = signal.spectrogram(data_new, fs=500, nperseg=256, noverlap=250)
        fig.add_heatmap(x=t, y=f, z=10 * np.log10(Sxx), colorscale='Viridis')

        fig.update_layout(
            title='Spectrogram',
            xaxis_title='Time [s]',
            yaxis_title='Frequency [Hz]',
        )

        st.plotly_chart(fig)

        st.markdown("### Spectrogram Explanation:")
        st.write("The spectrogram provides a visual representation of the power spectral density over time and frequency.")
        st.write("It helps visualize how different frequencies contribute to the signal, useful for identifying patterns.")

        # Alpha Power Over Time
        position_vector = np.where((f >= 8) & (f <= 13))[0]

        AlphaRange = np.mean(Sxx[position_vector[0]:position_vector[-1] + 1, :], axis=0)

        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Scatter(x=t, y=smooth_triangle(AlphaRange, 100), mode='lines', name='Alpha Power'))

        fig_alpha.update_layout(
            title='Alpha Power Over Time',
            xaxis_title='Time [s]',
            yaxis_title='Alpha Power',
        )

        st.plotly_chart(fig_alpha)

        st.markdown("### Alpha Power Over Time Explanation:")
        st.write("This plot shows the variation in alpha power (8-10 Hz) over time after smoothing.")
        st.write("Alpha waves are commonly associated with relaxed states.")

        # Statistical Analysis
        tg = np.array([4.2552, 14.9426, 23.2801, 36.0951, 45.4738, 59.3751, 72.0337, 85.0831, max(t) + 1])
        eyesclosed = []
        eyesopen = []
        j = 0  # Initial variable to traverse tg
        l = 0  # Initial variable to loop through the "y" data
        for i in range(0, len(t)):
            if t[i] >= tg[j]:
                if j % 2 == 0:
                    eyesopen.append(np.mean(AlphaRange[l:i]))
                if j % 2 == 1:
                    eyesclosed.append(np.mean(AlphaRange[l:i]))
                l = i
                j = j + 1

        fig_stat = go.Figure()
        fig_stat.add_box(y=eyesopen, name='Eyes Open', boxpoints='all', jitter=0.3, pointpos=-1.8)
        fig_stat.add_box(y=eyesclosed, name='Eyes Closed', boxpoints='all', jitter=0.3, pointpos=-0.6)

        fig_stat.update_layout(
            title='Statistical Analysis (Alpha Waves)',
            xaxis_title='State',
            yaxis_title='Alpha Power',
        )

        st.plotly_chart(fig_stat)

        meanopen = np.mean(eyesopen)
        meanclosed = np.mean(eyesclosed)
        sdopen = np.std(eyesopen)
        sdclosed = np.std(eyesclosed)

        st.markdown("### Statistical Analysis Explanation:")
        st.write("The box plot visually compares the distribution of alpha power during eyes open and eyes closed states.")
        st.write("The t-test result indicates statistical significance in the difference.")

        st.write("Mean (Eyes Open):", meanopen)
        st.write("Mean (Eyes Closed):", meanclosed)
        st.write("Standard Deviation (Eyes Open):", sdopen)
        st.write("Standard Deviation (Eyes Closed):", sdclosed)

        result = stats.ttest_ind(eyesopen, eyesclosed, equal_var=False)
        st.write("T-Test Result:")
        st.write("t-Statistic:", result.statistic)
        st.write("p-Value:", result.pvalue)


        position_vector = np.where((f >= 13) & (f <= 20))[0]

        AlphaRange = np.mean(Sxx[position_vector[0]:position_vector[-1] + 1, :], axis=0)

        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Scatter(x=t, y=smooth_triangle(AlphaRange, 100), mode='lines', name='Alpha Power'))

        fig_alpha.update_layout(
            title='Beta Power Over Time(_Lower_)',
            xaxis_title='Time [s]',
            yaxis_title='Beta Power',
        )

        st.plotly_chart(fig_alpha)

        st.markdown("### Low Beta Power Over Time Explanation:")
        st.write("This plot shows the variation in Beta power (13-20 Hz) over time after smoothing.")
        st.write("Beta waves(Low) are commonly associated with problem solving.")

        position_vector = np.where((f >= 13) & (f <= 30))[0]

        AlphaRange = np.mean(Sxx[position_vector[0]:position_vector[-1] + 1, :], axis=0)

        fig_alpha = go.Figure()
        fig_alpha.add_trace(go.Scatter(x=t, y=smooth_triangle(AlphaRange, 100), mode='lines', name='Alpha Power'))

        fig_alpha.update_layout(
            title='Beta Power Over Time(_Higher_)',
            xaxis_title='Time [s]',
            yaxis_title='Beta Power',
        )

        st.plotly_chart(fig_alpha)

        st.markdown("### High Beta Power Over Time Explanation:")
        st.write("This plot shows the variation in beta power (20-30 Hz) over time after smoothing.")
        st.write("Beta waves(High) are commonly associated with concentration and stress.")



if __name__ == "__main__":
    main()