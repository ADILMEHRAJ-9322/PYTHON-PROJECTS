import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("C:\\Users\\user\\Desktop\\cgl24_files\\Expanded_data_with_more_features.csv")
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())

# Drop unnecessary column
df = df.drop("Unnamed: 0", axis=1)
print(df.head())

# Gender distribution
plt.figure(figsize=(5,5))
ax = sns.countplot(data=df, x="Gender")
ax.bar_label(ax.containers[0])
plt.show()

# Group by Parent Education and plot heatmap
gb = df.groupby("ParentEduc").agg({"MathScore": 'mean', "ReadingScore": "mean", "WritingScore": "mean"})
print(gb)
sns.heatmap(gb, annot=True)
plt.title("Relationship between parents' education and students' scores")
plt.show()

# Group by Parent Marital Status and plot heatmap
gb1 = df.groupby("ParentMaritalStatus").agg({"MathScore": 'mean', "ReadingScore": "mean", "WritingScore": "mean"})
print(gb1)
sns.heatmap(gb1, annot=True)
plt.title("Relationship between parents' marital status and students' scores")
plt.show()

# Boxplot for MathScore
sns.boxplot(data=df, x='MathScore')
plt.show()

# Ethnic group distribution
print(df["EthnicGroup"].unique())
groupA = df.loc[(df["EthnicGroup"] == "group A")].count()
groupB = df.loc[(df["EthnicGroup"] == "group B")].count()
groupC = df.loc[(df["EthnicGroup"] == "group C")].count()
groupD = df.loc[(df["EthnicGroup"] == "group D")].count()
groupE = df.loc[(df["EthnicGroup"] == "group E")].count()
l = ["group A", "group B", "group C", "group D", "group E"]
mlist = [groupA["EthnicGroup"], groupB["EthnicGroup"], groupC["EthnicGroup"], groupD["EthnicGroup"], groupE["EthnicGroup"]]
plt.title("DISTRIBUTION OF ETHNIC GROUPS")
plt.pie(mlist, labels=l, autopct="%1.2f%%")
plt.show()

Key Insights from the Project
Gender Distribution:
The dataset contains more female students than male students.

Parental Education Impact:
Students whose parents have higher education levels tend to score better in Math, Reading, and Writing. This suggests a positive correlation between parental education and student performance.

Parental Marital Status:
The marital status of parents shows negligible impact on students' academic scores.

Math Score Outliers:
The boxplot for Math scores indicates the presence of outliers, suggesting that some students perform significantly differently from the majority.

Ethnic Group Distribution:
The dataset includes students from five ethnic groups, with their proportions visualized in a pie chart. This helps understand the demographic spread in the data.




