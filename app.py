import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ✅ Load CSVs from data folder (correct path)
df_vax = pd.read_csv("data/daily-covid-19-vaccine-doses-administered-per-million-people.csv")
df_deaths = pd.read_csv("data/estimated-cumulative-excess-deaths-per-100000-people-during-covid-19.csv")

# ✅ Rename columns for clarity
df_vax.rename(columns={
    'COVID-19 doses (daily, 7-day average, per million people)': 'vax_per_million'
}, inplace=True)

df_deaths.rename(columns={
    'Cumulative excess deaths per 100,000 people (central estimate)': 'excess_deaths_per_100k'
}, inplace=True)

# ✅ Convert date columns
df_vax['Day'] = pd.to_datetime(df_vax['Day'])
df_deaths['Day'] = pd.to_datetime(df_deaths['Day'])

# ✅ Merge datasets
df = pd.merge(df_vax, df_deaths, on=['Entity', 'Day'], how='inner')
df.dropna(subset=['vax_per_million', 'excess_deaths_per_100k'], inplace=True)

# ✅ Sidebar: choose country
countries = df['Entity'].unique()
selected_country = st.sidebar.selectbox("Select a Country", sorted(countries))

# ✅ Filter data for selected country
country_data = df[df['Entity'] == selected_country]

# ✅ App Title
st.title(f"{selected_country}: Vaccination vs Excess Deaths")

# ✅ Plot time series
st.subheader("Time Series Plot")
fig1, ax1 = plt.subplots()
ax1.plot(country_data['Day'], country_data['vax_per_million'], label='Vaccinations/million', color='blue')
ax1.plot(country_data['Day'], country_data['excess_deaths_per_100k'], label='Excess deaths/100k', color='red')
ax1.legend()
ax1.set_xlabel("Date")
ax1.set_ylabel("Rate")
ax1.grid(True)
st.pyplot(fig1)

# ✅ Regression Analysis
st.subheader("Linear Regression")
X = country_data[['vax_per_million']].values
y = country_data['excess_deaths_per_100k'].values

if len(X) > 1:
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y, alpha=0.5, label='Data')
    ax2.plot(X, model.predict(X), color='red', label='Regression Line')
    ax2.set_xlabel("Vaccinations/million")
    ax2.set_ylabel("Excess deaths/100k")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.markdown(f"**Slope**: {slope:.4f}")
    st.markdown(f"**Intercept**: {intercept:.4f}")
    st.markdown(f"**R² Score**: {r_squared:.4f}")
else:
    st.warning("Not enough data for regression.")
