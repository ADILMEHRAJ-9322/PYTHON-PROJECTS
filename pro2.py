import pandas as  pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
df=pd.read_csv("C:\\Users\\user\\Desktop\\cgl24_files\\Expanded_data_with_more_features.csv")
print(df.head())
print(df.describe())
print(df.info())
print(df.isnull().sum())
#DROP UNNAMED COLUMN
df=df.drop("Unnamed: 0",axis=1)
print(df.head())
#CHANGE WEEKLY STUDY HOURS COLUMN
#df[" WklyStudyHours "]=df[" WklyStudyHours "].str.replace("05-oct","5-10")
#gender distribution
plt.figure(figsize=(5,5))
ax=sns.countplot(data=df,x="Gender")
ax.bar_label(ax.containers[0])
plt.show()
#from the above chart we have analysed that the no of females in the data are more than no of males

gb=df.groupby("ParentEduc").agg({"MathScore": 'mean',"ReadingScore":"mean","WritingScore":"mean"})
print(gb)
sns.heatmap(gb,annot=True)
plt.title("relationship between parents education and students score")
plt.show()
# from this above chart we have concluded that the education of the parents have a good impact on their scores.
gb1=df.groupby("ParentMaritalStatus").agg({"MathScore": 'mean',"ReadingScore":"mean","WritingScore":"mean"})
print(gb1)
sns.heatmap(gb1,annot=True)
plt.title("relationship between parents marital status and students score")
plt.show()
# from the above chart we have concluded that there is negligible impact on the students score due to their parents marital status.
sns.boxplot(data=df,x='MathScore')
plt.show()
#box plot shows the outliers:
print(df["EthnicGroup"].unique())
#distribution of ethnic group:
groupA=df.loc[(df["EthnicGroup"]=="group A")].count()
groupB=df.loc[(df["EthnicGroup"]=="group B")].count()
groupC=df.loc[(df["EthnicGroup"]=="group C")].count()
groupD=df.loc[(df["EthnicGroup"]=="group D")].count()
groupE=df.loc[(df["EthnicGroup"]=="group E")].count()
l=["group A","group B","group C","group D","group E"]

mlist=[groupA["EthnicGroup"],groupB["EthnicGroup"],groupC["EthnicGroup"],groupD["EthnicGroup"],groupE["EthnicGroup"]]
plt.title("DISTRIBUTION OF ETHNIC GROUPS")
plt.pie(mlist,labels=l,autopct="%1.2f%%")
plt.show()




