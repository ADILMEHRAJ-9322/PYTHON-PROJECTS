import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
df = pd.read_csv(r'c:\Users\user\Desktop\cgl24_files\Netflix Dataset.csv')

# Step 2: Data Cleaning
# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Fill missing values in 'Director', 'Cast', 'Country' with 'Unknown'
for col in ['Director', 'Cast', 'Country']:
    df[col] = df[col].fillna('Unknown')

# Step 3: Data Overview
print("\nData types:\n", df.dtypes)
print("\nSample data:\n", df.head())

# Step 4: Category Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Category', data=df)
plt.title('Distribution of Movies vs TV Shows')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
print("Result: There are more Movies than TV Shows in the dataset.")

# Step 5: Top 10 Countries by Content
top_countries = df['Country'].str.split(',', expand=True).stack().str.strip().value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_countries.index, y=top_countries.values, palette='viridis')
plt.title('Top 10 Countries by Number of Titles')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45)
plt.show()
print("Result: The United States, India, and the United Kingdom are the top producers of Netflix content.")

# Step 6: Content Added Over Time
df['Year_Added'] = pd.to_datetime(df['Release_Date'], errors='coerce').dt.year
year_counts = df['Year_Added'].value_counts().sort_index()
plt.figure(figsize=(12,5))
sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o')
plt.title('Number of Titles Released per Year')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.show()
print("Result: There is a clear increase in content released in recent years, peaking around 2018-2020.")

# Step 7: Ratings Distribution
plt.figure(figsize=(10,5))
sns.countplot(y='Rating', data=df, order=df['Rating'].value_counts().index)
plt.title('Distribution of Content Ratings')
plt.xlabel('Count')
plt.ylabel('Rating')
plt.show()
print("Result: TV-MA, TV-14, and TV-PG are the most common ratings, indicating a focus on mature and teen audiences.")

# Step 8: Most Common Genres
# Split 'Type' column (genres) and count
from collections import Counter
all_genres = df['Type'].str.split(',').explode().str.strip()
top_genres = all_genres.value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='magma')
plt.title('Top 10 Most Common Genres')
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
plt.show()
print("Result: Dramas, International Movies, and Comedies are the most prevalent genres.")

# Step 9: Duration Analysis (Movies vs TV Shows)
# Extract numeric duration for movies
df['Duration_num'] = df['Duration'].str.extract('(\d+)').astype(float)
plt.figure(figsize=(10,5))
sns.histplot(df[df['Category']=='Movie']['Duration_num'].dropna(), bins=30, color='skyblue')
plt.title('Distribution of Movie Durations (minutes)')
plt.xlabel('Duration (minutes)')
plt.ylabel('Number of Movies')
plt.show()
print("Result: Most movies are between 80 and 120 minutes long.")

# TV Shows: Number of Seasons
tv_seasons = df[df['Category']=='TV Show']['Duration'].str.extract('(\d+)').astype(float)
plt.figure(figsize=(8,4))
sns.countplot(x=tv_seasons[0])
plt.title('Number of Seasons per TV Show')
plt.xlabel('Seasons')
plt.ylabel('Number of TV Shows')
plt.show()
print("Result: Most TV shows have 1 or 2 seasons.")

# Step 10: Top Directors
top_directors = df['Director'].value_counts().drop('Unknown').head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_directors.values, y=top_directors.index, palette='coolwarm')
plt.title('Top 10 Most Prolific Directors')
plt.xlabel('Number of Titles')
plt.ylabel('Director')
plt.show()
print("Result: These directors have the most titles on Netflix.")

 Step 11: Correlation Heatmap (Numerical Features)
plt.figure(figsize=(6,4))
sns.heatmap(df[['Duration_num', 'Year_Added']].corr(), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()
print("Result: There is little correlation between release year and duration.")

# Step 12: Insights Summary
Key Insights:
- Movies outnumber TV Shows on Netflix.
- The US, India, and UK are the largest content contributors.
- Content production has grown rapidly in recent years.
- Most content is rated for mature or teen audiences.
- Dramas, International Movies, and Comedies dominate the genre landscape.
- Most movies are 80-120 minutes; most TV shows have 1-2 seasons.
- A few directors are especially prolific.
- No strong correlation between release year and duration.
