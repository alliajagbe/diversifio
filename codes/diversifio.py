from scipy.optimize import minimize
import random
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

filepath = "beautiful_dataset.csv"
df = pd.read_csv(filepath)

features_to_keep = df.columns[((df.isnull().sum()/len(df))*100 < 5)].to_list()

df = df[features_to_keep]

df = df.dropna(subset=['grade'])
df['grade'] = pd.Categorical(
    df['grade'], categories=sorted(df['grade'].unique()))

df = df[df['deviation'] < 0.9]
### FROM HERE ##########################################

# Assuming your DataFrame is named 'df'
# Replace 'deviation' with the actual column name in your DataFrame
deviation_column = 'deviation'

# Check if 'deviation' column exists in the DataFrame
if deviation_column not in df.columns:
    raise KeyError(f"Column '{deviation_column}' not found in the DataFrame.")

# Initialize an empty DataFrame to store the selected rows
selected_rows = pd.DataFrame()

# Add the first four rows randomly from the dataset such that deviation < 0.009 and from either A or B
filtered_rows = df[(df[deviation_column] < 0.01) & (
    (df['grade'] == 'A') | (df['grade'] == 'B') )]
# selected_rows = selected_rows.append(filtered_rows.sample(n=3))
selected_rows = pd.concat([selected_rows, filtered_rows.sample(n=3)])


# Iterate until the difference in std is at least 0.05
while True:
    # Randomly select a row
    random_row = df.sample(n=1)

    # Check if there's at least one row in the selected_rows DataFrame
    if not selected_rows.empty:
        # Calculate the difference in std between the new row and the last one
        std_diff = abs(random_row[deviation_column] -
                       selected_rows[deviation_column].std())

        # Check if the difference is at least 0.05
        print(std_diff.values[0])
        if std_diff.values[0] >= 0.3:
            # selected_rows = selected_rows.append(random_row)
            selected_rows = pd.concat([selected_rows, random_row])
        else:
            continue
    else:
        # If selected_rows is empty, add the first randomly selected row
        selected_rows = selected_rows.append(random_row)

    # Break the loop if 10 rows are selected or another exit condition is met
    if len(selected_rows) == 12:
        break

# Display the selected rows
print(selected_rows)
df = selected_rows

### END ##########################################


def diversification_model(investor_amount, borrower_dict, risk_free_rate):
    df = pd.DataFrame(list(borrower_dict.values()), columns=[
                      'grade', 'term', 'loan_amnt', 'int_rate', 'deviation'])

    # Convert interest rate to numeric
    if df['int_rate'].dtype != 'float64':
        # Convert 'int_rate' to numeric
        df['int_rate'] = pd.to_numeric(
            df['int_rate'].str.rstrip('%'), errors='coerce')

    df['deviation'] = (1 - df['deviation']) * 100

    # Map grade to a numerical value (you may adjust the mapping based on the specific grading system)
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_numeric'] = df['grade'].map(grade_mapping)

    # Map term to a numerical value (shorter term is considered less risky)
    term_mapping = {'36 months': 1, '60 months': 2}
    df['term_numeric'] = df['term'].map(term_mapping)

    all_returns = []
    all_sharpe_ratios = []

    def objective(weights, df, risk_free_rate):
        portfolio_return = (weights * df['int_rate']).sum()
        portfolio_volatility = np.sqrt(
            ((weights ** 2) * (df['deviation']**2)).sum())
        sharpe_ratio = (portfolio_return - risk_free_rate) / \
            portfolio_volatility
        all_returns.append(portfolio_return)
        all_sharpe_ratios.append(sharpe_ratio)
        return -sharpe_ratio  # Negate to maximize

    def constraint(weights, investor_amount, df):
        return investor_amount - (weights * df['loan_amnt']).sum()

    # Define initial weights
    initial_weights = [1.0 / len(df)] * len(df)

    # Define bounds for each weight
    bounds = [(0, 1)] * len(df)

    # Set up optimization problem
    constraints = [{'type': 'eq', 'fun': constraint,
                    'args': (investor_amount, df)}]

    # Additional constraint: sum of weights for each grade should be less than or equal to 1
    grade_constraints = [{'type': 'ineq', 'fun': lambda weights, df=df, grade=grade: 1 - (weights[df['grade_numeric'] == grade].sum())}
                         for grade in grade_mapping.values()]

    # Additional constraint: total weight for each term should be less than or equal to 1
    term_constraints = [{'type': 'ineq', 'fun': lambda weights, df=df, term=term: 1 - (weights[df['term_numeric'] == term].sum())}
                        for term in term_mapping.values()]

    result = minimize(objective, initial_weights, args=(df, risk_free_rate), method='SLSQP',
                      bounds=bounds, constraints=constraints+grade_constraints+term_constraints)

    # Extract optimized weights
    optimized_weights = result.x

    # Calculate the proportions to be given to each borrower
    proportions = optimized_weights / optimized_weights.sum()

    # Calculate the expected return for the investor
    expected_return = (optimized_weights * df['int_rate']).sum()

    return proportions, expected_return, all_returns, all_sharpe_ratios


st.title("Welcome to Diversifio!, Ms. Khushi Goel!")

columns = st.columns(3)

with columns[0]:
    investor_amount = st.number_input(
        'Amount to be invested', value=10000, step=500)
with columns[1]:
    num_borrowers = st.number_input(
        'Number of borrowers to be selected', value=10, step=1)
with columns[2]:
    risk_free_rate = st.number_input('Risk-free rate', value=0.06, step=0.005)

borrower_dict = df[['grade', 'term', 'loan_amnt', 'int_rate',
                    'deviation']].sample(n=num_borrowers).to_dict(orient='index')

# Call the diversification_model function
proportions, expected_return, all_returns, all_sharpe_ratios = diversification_model(
    investor_amount, borrower_dict, risk_free_rate)

# Display Expected Return
st.subheader(f"Expected Return: {expected_return:.2f}%")

df_proportions = pd.DataFrame(proportions, columns=['Proportions'])
fig_proportions = px.pie(df_proportions, values='Proportions',
                         names=df_proportions.index, title='Proportions of Investment')
st.plotly_chart(fig_proportions)


st.write("Borrowers Information")
st.write(pd.DataFrame(borrower_dict).T)

# Visualize Returns during Optimization with Plotly
df_all_returns = pd.DataFrame({'Returns': all_returns})
fig_all_returns = px.line(df_all_returns, y='Returns',
                          title='Returns During Optimization', labels={'index': 'Iteration'})
st.plotly_chart(fig_all_returns)

# Visualize Sharpe Ratios during Optimization with Plotly
df_all_sharpe_ratios = pd.DataFrame({'Sharpe Ratios': all_sharpe_ratios})
fig_all_sharpe_ratios = px.line(df_all_sharpe_ratios, y='Sharpe Ratios',
                                title='Sharpe Ratios During Optimization', labels={'index': 'Iteration'})
st.plotly_chart(fig_all_sharpe_ratios)

# visualizing returns and sharpe ratios
df_all_returns_sharpe_ratios = pd.DataFrame(
    {'Returns': all_returns, 'Sharpe Ratios': all_sharpe_ratios})
fig_all_returns_sharpe_ratios = px.scatter(
    df_all_returns_sharpe_ratios, x='Returns', y='Sharpe Ratios', title='Returns vs. Sharpe Ratios During Optimization')
st.plotly_chart(fig_all_returns_sharpe_ratios)
