import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import csv

# === Step 1: Load and Inspect Data ===

print("Current Working Directory:", os.getcwd())
print("Files in data/:", os.listdir('../data'))

# Load vaccination and excess death datasets
vaccination_data = pd.read_csv('../data/daily-covid-19-vaccine-doses-administered-per-million-people.csv')
death_data = pd.read_csv('../data/estimated-cumulative-excess-deaths-per-100000-people-during-covid-19.csv')

# Rename long column names for clarity
vaccination_data.rename(columns={
    'COVID-19 doses (daily, 7-day average, per million people)': 'vax_per_million'
}, inplace=True)

death_data.rename(columns={
    'Cumulative excess deaths per 100,000 people (central estimate)': 'excess_deaths_per_100k',
    'Total confirmed deaths due to COVID-19 per 100,000 people': 'covid_deaths_per_100k'
}, inplace=True)

# Convert date columns to datetime format
vaccination_data['Day'] = pd.to_datetime(vaccination_data['Day'])
death_data['Day'] = pd.to_datetime(death_data['Day'])

# Merge datasets on country (Entity) and date
merged_data = pd.merge(vaccination_data, death_data, on=['Entity', 'Day'], how='inner')

# Check for missing values
print("\nMissing data summary:\n", merged_data.isnull().sum())

# Drop rows missing key columns
merged_data.dropna(subset=['vax_per_million', 'excess_deaths_per_100k'], inplace=True)

# === Step 2: Analyze Selected Countries ===

countries_to_analyze = ['India', 'United States', 'Brazil', 'Germany', 'Bangladesh']

# Store regression results for export
regression_results = []

for country in countries_to_analyze:
    print(f"\n===== Analyzing {country.upper()} =====")

    # Filter data for the current country
    country_data = merged_data[merged_data['Entity'] == country]

    if len(country_data) < 10:
        print(f"Insufficient data for {country}, skipping...")
        continue

    # === Plot vaccination and death trends over time ===
    plt.figure(figsize=(12, 6))
    plt.plot(country_data['Day'], country_data['vax_per_million'], label='Vaccinations per million', color='blue')
    plt.plot(country_data['Day'], country_data['excess_deaths_per_100k'], label='Excess deaths per 100k', color='red')
    plt.title(f'{country} - Vaccination and Excess Deaths Over Time')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Run linear regression: Vaccinations vs. Excess Deaths ===
    X = country_data[['vax_per_million']].values
    y = country_data['excess_deaths_per_100k'].values

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)

    # Print regression summary
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R² Score: {r_squared:.4f}")

    # Save results for export
    regression_results.append([country, slope, intercept, r_squared])

    # === Plot regression line over data ===
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label=f'{country} Data')
    plt.plot(X, model.predict(X), color='red', label='Regression Line')
    plt.xlabel('Vaccinations per million (7-day avg)')
    plt.ylabel('Excess deaths per 100k')
    plt.title(f'{country} - Regression Analysis')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Step 3: Export All Results to CSV ===

output_file = 'regression_results.csv'

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Country', 'Slope', 'Intercept', 'R_squared'])
    writer.writerows(regression_results)

print(f"\n Regression results saved to '{output_file}'")
