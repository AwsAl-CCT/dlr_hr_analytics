import streamlit as st
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.gridspec import GridSpec


@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/AwsAl-CCT/dlr_hr_analytics/refs/heads/main/QVal.csv'
    response = requests.get(url)
    url_headcount = 'https://raw.githubusercontent.com/AwsAl-CCT/dlr_hr_analytics/refs/heads/main/headcount_with_leaves.csv'
    response_headcount = requests.get(url_headcount)
    df = pd.read_csv(StringIO(response.text), encoding='utf-8', sep=',')
    df_headcount = pd.read_csv(url_headcount, encoding='utf-8')
    url_leave = "https://github.com/AwsAl-CCT/dlr_hr_analytics/raw/main/Person%20Balances.xlsx"
    df_leave = pd.read_excel(url_leave, engine="openpyxl")
    return df, df_headcount, df_leave

df, df_headcount, df_leave = load_data()

from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.gridspec import GridSpec

# Clean column names
df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True)

# Convert hourly rate to numeric
df['Avg. Hourly Rate of Pay'] = df['Hourly Rate of Pay'].replace('[^\\d.]', '', regex=True).astype(float)

# Convert 'Date Joined (Person)' to datetime and extract year
df['Date Joined'] = pd.to_datetime(df['Date Joined'], errors='coerce', dayfirst=True)
df['Year Joined'] = df['Date Joined'].dt.year

# Sort age and service ranges


age_order = sorted(
    [str(r) for r in df['Age Range'].unique()],
    key=lambda r: int(r.split('-')[0].strip()) if '-' in r else 999
)

service_order = sorted([str(x) for x in df['Length of Service Range'].dropna().unique()], key=lambda x: int(x.split('-')[0]) if '-' in x else 100)


# Irish language proficiency data
irish_data = {
    "Directorate": [
        "ARCHITECTS", "COMMUNITY, CULTURAL SERVICES & PARKS", "CORPORATE AFFAIRS",
        "FINANCE & WATER SERVICES", "FORWARD PLANNING INFRASTRUCTURE", "HOUSING",
        "INFRASTRUCTURE & CLIMATE CHANGE", "LAW", "PLANNING & ECONOMIC DEVELOPMENT"
    ],
    "No Irish": [31, 264, 135, 100, 12, 158, 336, 13, 151],
    "Some Irish": [0, 9, 6, 2, 0, 6, 7, 0, 6],
    "Level 3+": [1, 19, 7, 5, 0, 5, 8, 1, 9]
}
irish_df = pd.DataFrame(irish_data)
st.set_page_config(layout="wide")

# Set tabs
tab1, tab2, tab3, tab4= st.tabs(["📊 HR Dashboard", "🗣️ Irish Language Proficiency", "👥 Headcount", "🏖️ Leave Analytics"])


with tab1:
    st.title("DLR HR Analytics")
    st.write("Main Dashboard for HR")

    # -----------------------------
    # LAYOUT: 2 x 3 + 1 wide
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    # -----------------------------
    # 1. Gender Distribution
    # -----------------------------
    with col1:
        fig_gender = px.histogram(
            df,
            x="Gender",
            color="Gender",
            template="plotly_white",
            title="Gender Distribution"
        )
        fig_gender.update_layout(showlegend=False)
        st.plotly_chart(fig_gender, use_container_width=True)

    # -----------------------------
    # 2. Hourly Rate by Category (BOX) - RESET
    # -----------------------------
    with col2:
        fig_box = px.box(
            df,
            x="Category",
            y="Avg. Hourly Rate of Pay",
            color="Category",
            template="plotly_white",
            title="Hourly Rate by Category"
        )
        fig_box.update_layout(showlegend=False)
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------
    # 3. Employee Status Distribution
    # -----------------------------
    with col3:
        fig_status = px.histogram(
            df,
            x="Employee Status",
            color="Employee Status",
            template="plotly_white",
            title="Employee Status Distribution"
        )
        fig_status.update_layout(showlegend=False)
        fig_status.update_xaxes(tickangle=45)
        st.plotly_chart(fig_status, use_container_width=True)

    # -----------------------------
    # 4. Hourly Rate by Age Range (VIOLIN) - RESET
    # -----------------------------
    with col4:
        fig_violin = px.violin(
            df,
            x="Age Range",
            y="Avg. Hourly Rate of Pay",
            color="Age Range",
            category_orders={"Age Range": age_order},
            template="plotly_white",
            title="Hourly Rate by Age Range",
            points=False           # no dots
            # box=True REMOVED
        )

        fig_violin.update_traces(
            scalemode="width",     # ✅ Seaborn-like width
            width=0.8              # ✅ fuller violins
        )

        fig_violin.update_layout(showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)


    # -----------------------------
    # 5. Length of Service Distribution
    # -----------------------------
    with col5:
        fig_service = px.histogram(
            df,
            x="Length of Service Range",
            color="Length of Service Range",
            category_orders={"Length of Service Range": service_order},
            template="plotly_white",
            title="Length of Service Distribution"
        )
        fig_service.update_layout(showlegend=False)
        fig_service.update_xaxes(tickangle=45)
        st.plotly_chart(fig_service, use_container_width=True)

    # -----------------------------
    # 6. Total Employee Count by Category
    # -----------------------------
    with col6:
        fig_category = px.histogram(
            df,
            x="Category",
            color="Category",
            template="plotly_white",
            title="Total Employee Count by Category"
        )
        fig_category.update_layout(showlegend=False)
        fig_category.update_xaxes(tickangle=45)
        st.plotly_chart(fig_category, use_container_width=True)

    # -----------------------------
    # 7. Employees Joined Per Year (5-year ticks)
    # -----------------------------
    st.markdown("---")

    yearly_joined = (
        df.groupby("Year Joined")
          .size()
          .reset_index(name="Count")
          .sort_values("Year Joined")
    )

    fig_line = px.line(
        yearly_joined,
        x="Year Joined",
        y="Count",
        markers=True,
        template="plotly_white",
        title="Employees Joined Per Year"
    )
    fig_line.update_layout(showlegend=False)
    fig_line.update_xaxes(tickmode="linear", dtick=5)
    st.plotly_chart(fig_line, use_container_width=True)


with tab2:
    st.title("DLR Irish Language Proficiency")
    st.write("Org Level Proficiency")

    # Irish language proficiency data
    irish_data = {
        "Directorate": [
            "ARCHITECTS", "COMMUNITY, CULTURAL SERVICES & PARKS", "CORPORATE AFFAIRS",
            "FINANCE & WATER SERVICES", "FORWARD PLANNING INFRASTRUCTURE", "HOUSING",
            "INFRASTRUCTURE & CLIMATE CHANGE", "LAW", "PLANNING & ECONOMIC DEVELOPMENT"
        ],
        "No Irish": [31, 264, 135, 100, 12, 158, 336, 13, 151],
        "Some Irish": [0, 9, 6, 2, 0, 6, 7, 0, 6],
        "Level 3+": [1, 19, 7, 5, 0, 5, 8, 1, 9]
    }
    irish_df = pd.DataFrame(irish_data)

    # Totals and percentages
    irish_df["Total"] = irish_df["No Irish"] + irish_df["Some Irish"] + irish_df["Level 3+"]
    irish_df["% Level 3+"] = (irish_df["Level 3+"] / irish_df["Total"] * 100).round(0).astype(int)
    irish_df = irish_df.sort_values("Total", ascending=False)

    # Custom colors
    color_no_irish = "#4D9A94"      # red
    color_some_irish = "#D8B6A4"    # orange
    color_level3 = "#F28C28"        # green

    # --- Vertical summary chart ---
    summary_totals = {
        "No Irish": irish_df["No Irish"].sum(),
        "Some Irish": irish_df["Some Irish"].sum(),
        "Level 3+": irish_df["Level 3+"].sum()
    }
    total_all = sum(summary_totals.values())
    summary_percentages = {k: int(round(v / total_all * 100)) for k, v in summary_totals.items()}

    # Sort by total descending
    sorted_keys = sorted(summary_totals, key=summary_totals.get, reverse=True)
    sorted_values = [summary_totals[k] for k in sorted_keys]
    sorted_colors = [color_no_irish if k == "No Irish" else color_some_irish if k == "Some Irish" else color_level3 for k in sorted_keys]
    sorted_percentages = [summary_percentages[k] for k in sorted_keys]

    fig_summary, ax_summary = plt.subplots(figsize=(12, 3))
    bars = ax_summary.bar(sorted_keys, sorted_values, color=sorted_colors)

    # Value labels
    for bar, value in zip(bars, sorted_values):
        ax_summary.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(value),
                        ha='center', va='bottom')

    # Percentages in top-right corner inside chart area, stacked vertically
    x_pos = len(sorted_keys) - 1.01
    y_start = max(sorted_values) * 1.15
    line_spacing = 0.08 * max(sorted_values)

    for i, (pct, label, color) in enumerate(zip(sorted_percentages, sorted_keys, sorted_colors)):
        y_pos = y_start - i * line_spacing
        ax_summary.text(x_pos, y_pos, f"{pct}%", color=color, ha='right', va='top', fontsize=10)
        ax_summary.text(x_pos + 0.1, y_pos, f"{label}", color='black', ha='left', va='top', fontsize=10)
    

    ax_summary.set_ylim(0, max(sorted_values) * 1.2)
    ax_summary.set_yticks([])
    ax_summary.set_xlabel("")
    ax_summary.set_ylabel("")
    ax_summary.set_title("")
    ax_summary.grid(False)

    st.pyplot(fig_summary)

    # --- Horizontal stacked chart ---
    st.write("Irish Language Proficiency by Department")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.barh(irish_df["Directorate"], irish_df["No Irish"], label="No Irish", color=color_no_irish)
    ax2.barh(irish_df["Directorate"], irish_df["Some Irish"],
             left=irish_df["No Irish"], label="Some Irish", color=color_some_irish)
    ax2.barh(irish_df["Directorate"], irish_df["Level 3+"],
             left=irish_df["No Irish"] + irish_df["Some Irish"], label="Level 3+", color=color_level3)

    for i, (total, pct) in enumerate(zip(irish_df["Total"], irish_df["% Level 3+"])):
        ax2.text(total + 2, i, f"{total} ({pct}%)", va='center')

    max_total = irish_df["Total"].max()
    ax2.set_xticks(range(0, max_total + 100, 100))
    ax2.grid(axis='y', linestyle='', linewidth=0)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_title("")
    # ax2.legend()

    st.pyplot(fig2)

with tab3:
    st.title("DLR Headcount")
    st.write("Overall Headcount Information")


    # --- Horizontal Filters ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        payGroup = st.multiselect(
            "Select Pay Group",
            options=df_headcount['Pay Group (Person)'].unique(),
            default=['SALARIES']
        )

    with col2:
        appStatus = st.multiselect(
            "Select Appointment Status",
            options=df_headcount['Appointment Status (Appointment)'].unique(),
            default=['CO']
        )

    with col3:
        EmpStatus = st.multiselect(
            "Select Employment Status",
            options=df_headcount['Employment Status (Person)'].unique(),
            default=['Live']
        )

    with col4:
        postType = st.multiselect(
            "Select Post Type",
            options=df_headcount['Post Type (Post Profile)'].unique(),
            default=['PW']
        )

    # --- Apply Filters ---
    filtered_df = df_headcount.copy()
    if payGroup:
        filtered_df = filtered_df[filtered_df['Pay Group (Person)'].isin(payGroup)]
    if appStatus:
        filtered_df = filtered_df[filtered_df['Appointment Status (Appointment)'].isin(appStatus)]
    if EmpStatus:
        filtered_df = filtered_df[filtered_df['Employment Status (Person)'].isin(EmpStatus)]
    if postType:
        filtered_df = filtered_df[filtered_df['Post Type (Post Profile)'].isin(postType)]



    # --- Sunburst Chart ---
    sunburst_df = filtered_df.groupby(['Directorate', 'Department', 'Grade'])['Employee Number (Person)'].count().reset_index()
    sunburst_df.rename(columns={'Employee Number (Person)': 'Headcount'}, inplace=True)

    # Add Grade Label for clarity
    sunburst_df['Grade Label'] = 'Grade ' + sunburst_df['Grade'].astype(str)

    # Calculate total headcount
    total_headcount = sunburst_df['Headcount'].sum()
    st.subheader(f"Total Headcount (Filtered): {total_headcount}")

    # Create Sunburst chart
    fig = px.sunburst(
        sunburst_df,
        path=['Directorate', 'Department', 'Grade Label'],
        values='Headcount',
        color='Directorate',
        title="Headcount Hierarchy",
        width=950,
        height=950
    )

    # Improve text readability
    fig.update_traces(
        textinfo='label+value',
        insidetextorientation='radial',  # Uniform orientation
        hovertemplate='<b>%{label}</b><br>Headcount: %{value}<extra></extra>'
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Download Button ---
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_headcount.csv",
        mime="text/csv"
    )

with tab4:
    st.title("Leave Analytics")
    st.write("Leave takers behaviour")

    # -------------------------------------------------------------------
    # DATA PREP
    # -------------------------------------------------------------------
    df = df_leave.copy()

    numeric_cols = ["Balance", "Taken", "Total Entitlement"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Balance"])

    # -------------------------------------------------------------------
    # GLOBAL FILTER (NO HEADER, NO SPACING)
    # -------------------------------------------------------------------
    filter_container = st.container()
    with filter_container:
        if "Directorate (Person)" in df.columns:
            dirs = ["All"] + sorted(df["Directorate (Person)"].dropna().unique().tolist())
            selected_dir = st.selectbox("", dirs, index=0)
        else:
            selected_dir = "All"

    df_filtered = df.copy()
    if selected_dir != "All":
        df_filtered = df_filtered[df_filtered["Directorate (Person)"] == selected_dir]


    # -------------------------------------------------------------------
    # KPI CARDS
    # -------------------------------------------------------------------
    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Average Leave Balance", f"{df_filtered['Balance'].mean():.1f} days")

    with colB:
        st.metric(
            "Average Leave Taken",
            f"{df_filtered['Taken'].mean():.1f} days" if "Taken" in df.columns else "N/A"
        )

    with colC:
        st.metric("Employees Included", f"{len(df_filtered)}")


    # -------------------------------------------------------------------
    # TOP ROW
    # -------------------------------------------------------------------
    col1, col2 = st.columns(2)

    # ===========================
    # HISTOGRAM
    # ===========================
    with col1:
        st.markdown("Leave balance distribution")

        NBINS = 20

        fig_hist = px.histogram(
            df_filtered,
            x="Balance",
            nbins=NBINS,
            labels={"Balance": "Leave Balance (days)", "count": "Employees count"},
            title=""
        )
        fig_hist.update_layout(template="plotly_white", bargap=0.05, margin=dict(t=10))

        # KDE overlay
        kde_fig, ax = plt.subplots()
        sns.kdeplot(x=df_filtered["Balance"], ax=ax)
        line = ax.lines[0]
        xs = line.get_xdata()
        ys = line.get_ydata()
        plt.close(kde_fig)

        bin_width = (df_filtered["Balance"].max() - df_filtered["Balance"].min()) / NBINS
        ys_scaled = ys * len(df_filtered) * bin_width

        fig_hist.add_trace(go.Scatter(
            x=xs, y=ys_scaled,
            mode="lines",
            line=dict(width=2),
            showlegend=False
        ))

        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


    # ===========================
    # SCATTER
    # ===========================
    with col2:
        st.markdown("Leave taken vs leave balance")

        if "Taken" in df.columns and "Directorate (Person)" in df.columns:

            fig_scatter = px.scatter(
                df_filtered,
                x="Taken",
                y="Balance",
                color="Directorate (Person)",
                opacity=0.85,
                trendline="ols",
                labels={"Taken": "Leave Taken (days)", "Balance": "Leave Balance (days)"},
                title=""
            )

            fig_scatter.update_layout(
                template="plotly_white",
                showlegend=False,
                margin=dict(t=10)
            )

            st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

        else:
            st.info("Scatter plot requires 'Taken' & 'Directorate (Person)' columns.")


    # -------------------------------------------------------------------
    # BOTTOM ROW — ALWAYS SHOW ALL DIRECTORATES
    # -------------------------------------------------------------------
    st.markdown("---")
    st.markdown("Remaining leave by directorate")

    if "Directorate (Person)" in df.columns:

        df_dir = (
            df.groupby("Directorate (Person)", as_index=False)["Balance"]
            .mean()
            .sort_values("Balance", ascending=True)
            .rename(columns={"Balance": "Avg Remaining"})
        )

        fig_bar = px.bar(
            df_dir,
            x="Avg Remaining",
            y="Directorate (Person)",
            orientation="h",
            labels={"Avg Remaining": "Remaining Leave (days)"},
            title=""
        )

        fig_bar.update_layout(
            template="plotly_white",
            margin=dict(t=10),
            yaxis_title=None  # ← REMOVE AXIS TITLE
        )

        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("Directorate column missing; bottom chart cannot be drawn.")
