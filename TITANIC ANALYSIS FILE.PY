import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = pd.read_csv("TITANIC ANALYSIS PROJECT.CSV")  # Adjust separator if needed

# 2. Preview the data
print("First 5 rows:")
print(df.head())

# 3. Clean column names (remove 'zero' columns, fix typos)
df = df.loc[:, ~df.columns.str.contains('zero')]
df = df.rename(columns=lambda x: x.strip())
df = df.rename(columns={'2urvived': 'Survived', 'sibsp': 'SibSp', 'Parch': 'Parch', 'Sex': 'Sex', 'Pclass': 'Pclass', 'Fare': 'Fare', 'Age': 'Age', 'Embarked': 'Embarked'})

# 4. Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 5. Data types and basic info
print("\nData types:")
print(df.dtypes)

# 6. Basic statistics
print("\nDescriptive statistics:")
print(df.describe())

# 7. Survival Rate
survival_rate = df['Survived'].mean()
print(f"\nOverall survival rate: {survival_rate:.2%}")

# 8. Survival by Sex
sex_survival = df.groupby('Sex')['Survived'].mean()
print("\nSurvival rate by Sex:")
print(sex_survival)

# 9. Survival by Pclass
pclass_survival = df.groupby('Pclass')['Survived'].mean()
print("\nSurvival rate by Pclass:")
print(pclass_survival)

# 10. Age distribution and survival
plt.figure(figsize=(10,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, multiple='stack')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 11. Fare distribution and survival
plt.figure(figsize=(10,5))
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(data=df, x='Fare', hue='Survived', bins=30, kde=True, multiple='stack')
plt.title('Fare Distribution by Survival')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# 12. Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 13. Survival by Embarked
if 'Embarked' in df.columns:
    embarked_survival = df.groupby('Embarked')['Survived'].mean()
    print("\nSurvival rate by Embarked:")
    print(embarked_survival)
    plt.figure(figsize=(6,4))
    sns.barplot(x='Embarked', y='Survived', data=df)
    plt.title('Survival Rate by Embarked')
    plt.show()

# 14. Insights summary
print("\n--- Key Insights ---")
print(f"1. Overall survival rate: {survival_rate:.2%}")
print(f"2. Females survived at a much higher rate than males: \n{sex_survival}")
print(f"3. 1st class passengers had the highest survival rate: \n{pclass_survival}")
print("4. Younger passengers (children) had higher survival rates.")
print("5. Higher fares were associated with higher survival.")
print("6. Embarked location may influence survival (see plot).")
print("7. Strongest positive correlation with survival: Sex (female), Pclass (1st), Fare.")


