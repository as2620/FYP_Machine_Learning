import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from the CSV file
# csv_filename = "breathing_rate_data.csv"
csv_filename = "box_data.csv"

csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

data = pd.read_csv(csv_filepath)

data= data[data["Participant ID"] != 10]

# class_order = ['3bpm', '5bpm', '10bpm', '15bpm', '20bpm', '30bpm', '40bpm']
class_order = ['Rest', 'Box']

# palette = sns.color_palette(["#e81416", "#ffa500", "#faeb36", "#79c314", "#36cedc" ,"#487de7", "#70369d"], len(class_order))
palette = sns.color_palette(["#ffa500", "#36cedc"], len(class_order))

plt.figure(figsize=(10, 6))  # Adjust figure size as needed
sns.boxplot(x="Participant ID", y="Average CO2 Exhaled", hue="Breathing Type", data=data, palette=palette)

plt.title("Box Plots for Breathing Data")
plt.xlabel("Participant ID")
plt.ylabel("Breathing Type")
plt.legend(title="Breathing Type")

plt.show()
