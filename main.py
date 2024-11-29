import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tpot import TPOTClassifier
import os
from pandas.plotting import scatter_matrix
import sweetviz as sv

df = pd.read_csv('processed_data\gym_exercise_dataset.csv')

print(df.info())
print(df.describe())
print(df.head())

numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

non_numeric_cols = df.select_dtypes(exclude=['number']).columns
for col in non_numeric_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

sns.boxplot(data=df['Difficulty (1-5)'])
plt.show()

scatter_matrix(df[numeric_cols], figsize=(10, 10))
plt.show()

eda_report = sv.analyze(df)
reports_dir = 'reports'
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

report_path = os.path.join(reports_dir, 'report.html')
eda_report.show_html(report_path)

print(f"Report generated at {report_path}")

df_features = df[numeric_cols].drop(columns=['target_column'])
df_target = df['target_column']

X_train = df_features
y_train = df_target

tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20)
tpot.fit(X_train, y_train)

tpot.export('best_model.py')

print("Model trained and exported!")
