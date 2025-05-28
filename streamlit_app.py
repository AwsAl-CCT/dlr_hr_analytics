import streamlit as st

import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns


url = 'https://raw.githubusercontent.com/AwsAl-CCT/dlr_hr_analytics/refs/heads/main/QVal.csv'
response = requests.get(url)
df = pd.read_csv(StringIO(response.text), encoding='utf-16', delimiter='\t')

from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.gridspec import GridSpec

# Clean column names
df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True)

# Convert hourly rate to numeric
df['Avg. Hourly Rate of Pay (Current Year)'] = df['Avg. Hourly Rate of Pay (Current Year)'].replace('[^\\d.]', '', regex=True).astype(float)

# Convert 'Date Joined (Person)' to datetime and extract year
df['Date Joined (Person)'] = pd.to_datetime(df['Date Joined (Person)'], errors='coerce', dayfirst=True)
df['Year Joined'] = df['Date Joined (Person)'].dt.year

# Sort age and service ranges
age_order = sorted(df['Age Range at Snapshot Date'].unique())
service_order = sorted(df['Length of Service Range at Snapshot Date'].unique(), key=lambda x: int(x.split('-')[0]) if '-' in x else 100)

# Set style
sns.set(style="whitegrid")

# Create custom grid layout
fig = plt.figure(figsize=(20, 18))
gs = GridSpec(3, 3, figure=fig)

# Define axes using the grid
axes = [
    fig.add_subplot(gs[0, 0]),  # Gender Distribution
    fig.add_subplot(gs[0, 1]),  # Hourly Rate by Category
    fig.add_subplot(gs[0, 2]),  # Employee Status Distribution
    fig.add_subplot(gs[1, 0]),  # Hourly Rate by Age Range
    fig.add_subplot(gs[1, 1]),  # Length of Service Distribution
    fig.add_subplot(gs[1, 2]),  # Employee Count by Category and Status
    fig.add_subplot(gs[2, :])   # Employees Joined Per Year (full width)
]

# Plot 1: Gender Distribution
sns.countplot(ax=axes[0], x="Gender (Person)", hue="Gender (Person)", data=df, palette="Set2", legend=False)
axes[0].set_title("Gender Distribution")
axes[0].set_xlabel("Gender")

# Plot 2: Boxplot of Hourly Rate by Employee Category
sns.boxplot(ax=axes[1], x="Category at Snapshot Date", y="Avg. Hourly Rate of Pay (Current Year)",
            hue="Category at Snapshot Date", data=df, palette="Set3", legend=False)
axes[1].set_title("Hourly Rate by Employee Category")
axes[1].set_xlabel("Employee Category")
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylabel("Avg. Hourly Rate of Pay")
axes[1].yaxis.set_major_locator(plt.MaxNLocator(5))

# Plot 3: Employee Status Distribution
sns.countplot(ax=axes[2], x="Employee Status at Snapshot Date", hue="Employee Status at Snapshot Date",
              data=df, palette="Set1", legend=False)
axes[2].set_title("Employee Status Distribution")
axes[2].set_xlabel("Employee Status")
axes[2].tick_params(axis='x', rotation=45)

# Plot 4: Violin Plot of Hourly Rate by Age Range
sns.violinplot(ax=axes[3], x="Age Range at Snapshot Date", y="Avg. Hourly Rate of Pay (Current Year)",
               hue="Age Range at Snapshot Date", data=df, palette="coolwarm", order=age_order, legend=False)
axes[3].set_title("Hourly Rate by Age Range")
axes[3].set_xlabel("Age Range")
axes[3].set_ylabel("Avg. Hourly Rate of Pay")
axes[3].yaxis.set_major_locator(plt.MaxNLocator(5))

# Plot 5: Length of Service Distribution
sns.countplot(ax=axes[4], x="Length of Service Range at Snapshot Date",
              hue="Length of Service Range at Snapshot Date", data=df,
              palette="Blues", order=service_order, legend=False)
axes[4].set_title("Length of Service Distribution")
axes[4].set_xlabel("Length of Service Range")
axes[4].tick_params(axis='x', rotation=45)

# Plot 6: Employee Count by Category and Status (log scale, readable labels, legend restored)
sns.countplot(ax=axes[5], x="Category at Snapshot Date", hue="Employee Status at Snapshot Date", data=df, palette="Set2")
axes[5].set_title("Employee Count by Category and Status")
axes[5].set_xlabel("Employee Category")
axes[5].tick_params(axis='x', rotation=45)
axes[5].set_yscale("log")
axes[5].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y):,}'))  # Human-readable
axes[5].legend(title="Employee Status", bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot 7: Employees Joined Per Year (line plot, 2000–2025, 5-year ticks)
yearly_joined = df['Year Joined'].value_counts().sort_index()
axes[6].plot(yearly_joined.index, yearly_joined.values, marker='o', linestyle='-', color='teal')
axes[6].set_title("Employees Joined Per Year")
axes[6].set_xlabel("Year")
axes[6].set_ylabel("Number of Employees")
axes[6].tick_params(axis='x', rotation=45)
axes[6].xaxis.set_major_locator(MultipleLocator(5))
axes[6].set_xlim(2000, 2025)

# Final layout
plt.tight_layout()



st.title("DLR HR Analytics")
st.write(
    "Main Dashboard for HR"
)

st.pyplot(fig)