import os
import skrf as rf
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for calculations
from datetime import datetime  # Import datetime for timestamp
from prettytable import PrettyTable  # Import for tabular data display

# Amateur radio bands with their names (in MHz)
amateur_bands = [
    (1.8, 2.0, "160m"),
    (3.5, 4.0, "80m"),
    (7.0, 7.3, "40m"),
    (10.1, 10.15, "30m"),
    (14.0, 14.35, "20m"),
    (18.068, 18.168, "17m"),
    (21.0, 21.45, "15m"),
    (24.89, 24.99, "12m"),
    (28.0, 29.7, "10m"),
    (50.0, 54.0, "6m"),
    (144.0, 148.0, "2m"),
    (420.0, 450.0, "70cm")
]

# Map bands to fixed colors
color_map = {
    "160m": "blue",
    "80m": "orange",
    "40m": "green",
    "30m": "red",
    "20m": "purple",
    "17m": "brown",
    "15m": "pink",
    "12m": "gray",
    "10m": "olive",
    "6m": "cyan",
    "2m": "magenta",
    "70cm": "yellow"
}

# Search for .s1p files in the current directory
s1p_files = [f for f in os.listdir('.') if f.endswith('.s1p')]

# Process each .s1p file
for filename in s1p_files:
    network = rf.Network(filename)
    file_base = os.path.splitext(filename)[0]  # Filename without extension

    # Extract the frequency range
    frequencies = network.f / 1e6  # Convert frequency to MHz
    min_freq = frequencies.min()
    max_freq = frequencies.max()

    # Filter amateur bands within the file's frequency range
    filtered_bands = [
        band for band in amateur_bands if band[0] <= max_freq and band[1] >= min_freq
    ]

    # Calculate VSWR and Log-Magnitude
    s11_magnitude = abs(network.s[:, 0, 0])  # S11 magnitude
    vswr = (1 + s11_magnitude) / (1 - s11_magnitude)
    s11_log_mag = 20 * np.log10(s11_magnitude)  # Convert to dB

    # Find minimum VSWR value and corresponding frequency
    min_vswr = vswr.min()
    min_vswr_freq = frequencies[np.argmin(vswr)]

    # Find the frequencies where VSWR crosses 2.0
    cross_indices = np.where(np.diff(np.sign(vswr - 2.0)))[0]
    lower_vswr_limit = frequencies[cross_indices[0]] if len(cross_indices) > 0 else None
    upper_vswr_limit = frequencies[cross_indices[-1] + 1] if len(cross_indices) > 1 else None

    # Calculate antenna bandwidth at 2.0 VSWR
    antenna_bw = (upper_vswr_limit - lower_vswr_limit) if lower_vswr_limit and upper_vswr_limit else None

    # Create figure with horizontal regions
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(file_base, fontsize=16)  # Use filename without extension as title

    # Left region: Smith Chart
    ax_smith = fig.add_axes([0.05, 0.1, 0.3, 0.8])  # (x, y, width, height)
    network.plot_s_smith(m=0, n=0, ax=ax_smith)
    ax_smith.set_title("Smith Chart")

    # Middle region: VSWR and Log-Mag Plot
    ax_vswr = fig.add_axes([0.4, 0.1, 0.4, 0.8])  # (x, y, width, height)
    ax_logmag = ax_vswr.twinx()

    # Add band overlays
    for band in filtered_bands:
        band_color = color_map[band[2]]
        ax_vswr.axvspan(
            max(band[0], min_freq),
            min(band[1], max_freq),
            color=band_color,
            alpha=0.3,
            label=f"{band[2]} ({band[0]}-{band[1]} MHz)"
        )

    vswr_line, = ax_vswr.plot(
        frequencies, vswr, label="VSWR", color="green", linewidth=1.5, linestyle="-"
    )
    log_mag_line, = ax_logmag.plot(
        frequencies, s11_log_mag, label="S11 Log-Mag (dB)", color="blue", linewidth=1.5, linestyle="-"
    )
    ax_vswr.axhline(2.0, color="red", linestyle="--", linewidth=2, label="VSWR = 2.0")
    ax_vswr.set_title("VSWR and Log-Mag vs Frequency")
    ax_vswr.set_xlabel("Frequency (MHz)")
    ax_vswr.set_ylabel("VSWR", color="green")
    ax_vswr.tick_params(axis="y", labelcolor="green")
    ax_vswr.grid(True)
    ax_logmag.set_ylabel("S11 Magnitude (dB)", color="blue")
    ax_logmag.tick_params(axis="y", labelcolor="blue")

    # Right region: Table and Legend
    ax_right = fig.add_axes([0.85, 0.1, 0.1, 0.8])  # (x, y, width, height)
    ax_right.axis("off")

    # Conditionally include the table
    if len(filtered_bands) <= 1:
        # Table at the top of the right region
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["Minimum VSWR", f"{min_vswr:.3f}"])
        table.add_row(["Min VSWR Frequency (MHz)", f"{min_vswr_freq:.3f}"])
        table.add_row(["Lower VSWR Limit (MHz)", f"{lower_vswr_limit:.3f}" if lower_vswr_limit else "N/A"])
        table.add_row(["Upper VSWR Limit (MHz)", f"{upper_vswr_limit:.3f}" if upper_vswr_limit else "N/A"])
        table.add_row(["Antenna BW at 2.0 VSWR (MHz)", f"{antenna_bw:.3f}" if antenna_bw else "N/A"])
        ax_right.text(0, 0.9, str(table), fontsize=10, family="monospace", va="top", ha="left", transform=ax_right.transAxes)

    # Legend below the table or at the top if no table
    legend_lines = [vswr_line, log_mag_line]
    legend_labels = ["VSWR", "S11 Log-Mag (dB)"]

    # Add bands to the legend
    for band in filtered_bands:
        band_color = color_map[band[2]]
        legend_lines.append(plt.Line2D([0], [0], color=band_color, lw=5))
        legend_labels.append(f"{band[2]} ({band[0]}-{band[1]} MHz)")

    ax_right.legend(
        legend_lines, legend_labels,
        loc="lower center" if len(filtered_bands) <= 1 else "upper center",
        fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.2 if len(filtered_bands) <= 1 else 0.9)
    )

    # Save the figure with a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    image_filename = f"{file_base}_{timestamp}.png"
    plt.savefig(image_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {image_filename}")

    # Close the figure to avoid displaying multiple plots in interactive environments
    plt.close(fig)
