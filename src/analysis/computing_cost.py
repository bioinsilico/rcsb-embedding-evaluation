import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--computing-time-file', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    out_path = args.out_path
    filename = args.computing_time_file

    # --- Load data ---
    data = pd.read_csv(filename, sep=r'\s+', header=None, names=['Sequence_Length', 'Computing_Time_ms'])

    # --- Remove outliers based on the IQR method ---
    Q1 = data['Computing_Time_ms'].quantile(0.01)
    Q3 = data['Computing_Time_ms'].quantile(0.99)
    IQR = Q3 - Q1

    # Define bounds (you may adjust the 1.5 multiplier if needed)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_data = data[
        (data['Computing_Time_ms'] >= lower_bound) &
        (data['Computing_Time_ms'] <= upper_bound) &
        (data['Sequence_Length'] <= 2500)
    ]

    # --- Quadratic polynomial fit ---
    x = filtered_data['Sequence_Length'].values
    y = filtered_data['Computing_Time_ms'].values
    coeffs = np.polyfit(x, y, deg=2)
    poly_eq = np.poly1d(coeffs)

    # --- Display coefficients ---
    print("Quadratic fit coefficients:")
    print(f"a (x² term): {coeffs[0]:.6e}")
    print(f"b (x term):  {coeffs[1]:.6e}")
    print(f"c (constant): {coeffs[2]:.6e}")

    # --- Generate fitted curve for plotting ---
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = poly_eq(x_fit)

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=20, alpha=0.7, label='Data')
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Quadratic fit')
    # plt.title('Single GPU A100 40GB')
    plt.title('10 CPU (i7-12700K) 64GB RAM')
    plt.xlabel('Sequence Length')
    plt.ylabel('Computing Time (ms)')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{out_path}/cpu-time.png", bbox_inches='tight', dpi=300)
    plt.show()