import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

trader_df = pd.read_csv('data/historical_data.csv')
sentiment_df = pd.read_csv('data/fear_greed.csv')

print(trader_df.head())
print(sentiment_df.head())

print(trader_df.info())
print(trader_df.describe())
print(trader_df.isnull().sum())

# Convert trader timestamp to datetime
trader_df['Timestamp'] = pd.to_datetime(
    trader_df['Timestamp'],
    unit='ms'
)

# Show converted timestamps
print(trader_df[['Timestamp']].head())


# Create date column from trader timestamp
trader_df['Date'] = trader_df['Timestamp'].dt.date

# Convert sentiment dataset date column
sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date

# Check output
print(trader_df[['Timestamp', 'Date']].head())

print(sentiment_df[['date', 'classification']].head())

# Merge datasets using Date
merged_df = pd.merge(
    trader_df,
    sentiment_df,
    left_on='Date',
    right_on='date',
    how='inner'
)

# Check merged data
print(merged_df.head())

# Shape of merged dataset
print("Merged Shape:", merged_df.shape)

# Average profit/loss by market sentiment
sentiment_pnl = merged_df.groupby(
    'classification'
)['Closed PnL'].mean()

print(sentiment_pnl)


# Create figure
plt.figure(figsize=(8, 5))

# Bar chart
sns.barplot(
    x=sentiment_pnl.index,
    y=sentiment_pnl.values
)

# Labels and title
plt.title('Average Trader PnL by Market Sentiment')
plt.xlabel('Market Sentiment')
plt.ylabel('Average Closed PnL')

# Show graph
# plt.show()

# Create win/loss column
merged_df['Win'] = merged_df['Closed PnL'] > 0

# Calculate win rate by sentiment
win_rate = merged_df.groupby(
    'classification'
)['Win'].mean() * 100

print(win_rate)

# Average trade size by sentiment
trade_size = merged_df.groupby(
    'classification'
)['Size USD'].mean()

print(trade_size)

# Top 10 traders by total profit
top_traders = merged_df.groupby(
    'Account'
)['Closed PnL'].sum().sort_values(
    ascending=False
).head(10)

print(top_traders)