import json
import matplotlib.pyplot as plt

# Load the profiling data from the JSON file
file_path = 'profiling_results.json'
with open(file_path, 'r') as file:
    profiling_data = json.load(file)

# Extract relevant data for analysis
functions = []
total_times = []
cumulative_times = []
total_calls = []

for func_name, stats in profiling_data.items():
    functions.append(func_name)
    total_times.append(stats['total_time'])
    cumulative_times.append(stats['cumulative_time'])
    total_calls.append(stats['total_calls'])

# Sort the data by cumulative time (descending)
sorted_indices = sorted(range(len(cumulative_times)), key=lambda i: cumulative_times[i], reverse=True)
functions_sorted = [functions[i] for i in sorted_indices][:10]  # Top 10 functions
cumulative_times_sorted = [cumulative_times[i] for i in sorted_indices][:10]
total_times_sorted = [total_times[i] for i in sorted_indices][:10]
total_calls_sorted = [total_calls[i] for i in sorted_indices][:10]

# Plotting the cumulative time for the top 10 functions
plt.figure(figsize=(10, 6))
plt.barh(functions_sorted, cumulative_times_sorted, color='skyblue')
plt.xlabel('Cumulative Time (seconds)')
plt.ylabel('Function')
plt.title('Top 10 Functions by Cumulative Time')
plt.gca().invert_yaxis()
plt.show()

# Plotting the number of calls for the top 10 functions
plt.figure(figsize=(10, 6))
plt.barh(functions_sorted, total_calls_sorted, color='lightgreen')
plt.xlabel('Number of Calls')
plt.ylabel('Function')
plt.title('Top 10 Functions by Number of Calls')
plt.gca().invert_yaxis()
plt.show()