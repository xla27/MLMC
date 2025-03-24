import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_arc(ax, center, width, height, angle_start, angle_end, color, label=None):
    t = np.linspace(np.radians(angle_start), np.radians(angle_end), 100)
    x = center[0] + width * np.cos(t)
    y = center[1] + height * np.sin(t)
    ax.plot(x, y, color=color, linewidth=3.5, label=label)

# Given angles and lengths
defl_angle1 = 15 * np.pi / 180
defl_angle2 = 45 * np.pi / 180
length1 = 0.2
length2 = 0.2
length3 = 0.15

# Calculate points based on the given lengths and angles
X1 = -0.05
X3 = length1 * np.cos(defl_angle1)
Y2 = length1 * np.sin(defl_angle1)
X4 = X3 + length2 * np.cos(defl_angle2)
Y4 = Y2 + length2 * np.sin(defl_angle2)
X5 = X4 + length3
Y6 = X5 - X1

# Define the points
points = {
    "Point 1": (X1, 0),
    "Point 2": (0, 0),
    "Point 3": (X3, Y2),
    "Point 4": (X4, Y4),
    "Point 5": (X5, Y4),
    "Point 6": (X5, Y6),
}

# Define the lines connecting the points with their respective colors and labels
lines = [
    ("Point 1", "Point 2", 'orange', '$\mathrm{Symmetry}$'),
    ("Point 2", "Point 3", 'green', '$\mathrm{Wall}$'),
    ("Point 3", "Point 4", 'green', '$\mathrm{Wall}$'),
    ("Point 4", "Point 5", 'green', '$\mathrm{Wall}$'),
    ("Point 5", "Point 6", 'purple', '$\mathrm{Supersonic} \, \mathrm{outlet}$'),
]

# Upper and lower spline coordinates
upper_x = np.array([
    -0.02020936303698075, 0, 0.03395543144123607, 0.09082846564336373, 0.14661820395592706, 0.173318,
    0.19319992720719353, 0.205, 0.22190726828064844, 0.24682307374062817, 0.2652391038632219,
    0.29882127643971634, 0.33890322435359677, 0.3611107900896657, 0.3919847229422493, 0.42123371196048637,
    0.4493994050891591, 0.48352322561043576
])

upper_y = np.array([
    0, 0.0125, 0.025504783059269658, 0.04775363636629213, 0.06891717975589887, 0.079515,
    0.09182479339325841, 0.11, 0.13603577480084267, 0.16876569215814605, 0.1899292355477528,
    0.23822552687275278, 0.28109526861067413, 0.3185384607615168, 0.35164041324269657, 0.3766025413432584,
    0.40536325415477525, 0.4373798967185393 
])

lower_x = np.array([
    0, 0.0675376040177305, 0.1319937094468085, 0.16070105052026346, 0.1829086162563323,
    0.18291, 0.1934, 0.211, 0.23599011484498483, 0.25819768058105375, 0.2804052463171226,
    0.30207116410840934, 0.3302368572370821, 0.3578609024209727, 0.39306801883181364,
    0.4271918393530902, 0.45589918042654515, 0.4846065215
])

lower_y = np.array([
    0, 0.01790761363735955, 0.04015646694438202, 0.05372284091207864, 0.06566125000365168,
    0.048296291325, 0.0517, 0.069, 0.0949646177738764, 0.13132250000730336, 0.15411400827303368,
    0.1866733057955056, 0.225201807863764, 0.2545051756339887, 0.28706447315646066,
    0.3207090805963483, 0.3467565186143258, 0.3710598863845505
])

x_range_upper = np.linspace(min(upper_x), max(upper_x), 100)
y_interp_upper = np.interp(x_range_upper, upper_x, upper_y)

x_range_lower = np.linspace(min(lower_x), max(lower_x), 100)
y_interp_lower = np.interp(x_range_lower, lower_x, lower_y)


# Set up the plot
# Set custom font parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 20 })


# Set up the plot with custom figure size
fig, ax = plt.subplots(figsize=(13, 11))

# Plot the lines
for line in lines:
    point1, point2, color, label = line
    x_values = [points[point1][0], points[point2][0]]
    y_values = [points[point1][1], points[point2][1]]
    ax.plot(x_values, y_values, color=color, linewidth=3.5, label=label)

# Plot arcs (farfield)
plot_arc(ax, (X5, 0), X5 - X1, X5 - X1, 90, 180, 'blue', '$\mathrm{Farfield}$')

# # Fill the area between the two curves
# polygon_x = np.concatenate([x_range_upper, np.flip(x_range_lower)])
# polygon_y = np.concatenate([y_interp_upper, np.flip(y_interp_lower)])
# ax.fill(polygon_x, polygon_y, color='lightgrey', label='Refinement region')

# Add labels for the axes
ax.set_xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
ax.set_ylabel(r'$y, \; \mathrm{m} $', fontsize=35)
ax.tick_params(labelsize=30)

# Add labels for L1, L2, and L3 near the wall lines, slightly below the lines
offset = -0.02  # Vertical offset to position the labels below the lines
ax.text((points["Point 2"][0] + points["Point 3"][0]) / 2 + 0.04, 
        (points["Point 2"][1] + points["Point 3"][1]) / 2 +0.004, 
        r'$L_1$', fontsize=30, ha='center', va='top')

ax.text((points["Point 3"][0] + points["Point 4"][0]) / 2 + 0.04, 
        (points["Point 3"][1] + points["Point 4"][1]) / 2 +0.02, 
        r'$L_2$', fontsize=30, ha='center', va='top')

ax.text((points["Point 4"][0] + points["Point 5"][0]) / 2, 
        (points["Point 4"][1] + points["Point 5"][1]) / 2 + offset/3, 
        r'$L_3$', fontsize=30, ha='center', va='top')

# Add angle labels for theta_1 and theta_2
import matplotlib.patches as patches

# Function to add an angle arc
def add_angle_arc(ax, center, radius, angle_start, angle_end, label, label_offset):
    arc = patches.Arc(center, radius*2, radius*2, angle=0, theta1=angle_start, theta2=angle_end, color='black', linewidth=1.5)
    ax.add_patch(arc)
    ax.text(center[0] + label_offset[0], center[1] + label_offset[1], label, fontsize=30, ha='center', va='bottom')

    # Add a dotted horizontal line from the center
    x_end = center[0] + radius * np.cos(np.radians(angle_start))
    y_end = center[1] + radius * np.sin(np.radians(angle_start))
    ax.plot([center[0], x_end], [center[1], y_end], linestyle=':', color='black')

# Add angle arcs for theta_1 and theta_2
add_angle_arc(ax, points["Point 2"], 0.06, 0, np.degrees(defl_angle1), r'$\vartheta_1$', (0.09, -0.01))
add_angle_arc(ax, points["Point 3"], 0.06, 0, np.degrees(defl_angle2), r'$\vartheta_2$', (0.08, 0.006))

# Add the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=24, loc='upper left', bbox_to_anchor=(-0.01, 1.01))

# Set the axes limits
ax.set_xlim(-0.185, 0.5)
ax.set_ylim(-0.01, 0.57)

# Set the aspect ratio to be equal
ax.set_aspect('equal')
ax.grid(True)

# Save
save_name = 'dw_geom.svg'
plt.savefig(save_name, bbox_inches='tight')


# Show the plot
plt.show()
