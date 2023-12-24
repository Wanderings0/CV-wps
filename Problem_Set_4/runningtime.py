import matplotlib.pyplot as plt

# Data
window_sizes = [17, 17, 17, 27, 27, 27, 37, 37, 37]
matching_functions = ['SSD', 'SAD', 'normalized_correlation', 'SSD', 'SAD', 'normalized_correlation', 'SSD', 'SAD', 'normalized_correlation']
running_times = [574.21, 749.43, 4971.3, 669.38, 656.83, 5271.06, 782.69, 762.18, 5557.66]

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot each matching function
for matching_function in set(matching_functions):
    times = [running_times[i] for i in range(len(running_times)) if matching_functions[i] == matching_function]
    sizes = [window_sizes[i] for i in range(len(window_sizes)) if matching_functions[i] == matching_function]
    ax.plot(sizes, times, label=matching_function)

# Set chart title and labels
ax.set_title('the running time of moebius')
ax.set_xlabel('Window Size')
ax.set_ylabel('Running Time (seconds)')

# Show legend
ax.legend()

# Display the plot
plt.show()