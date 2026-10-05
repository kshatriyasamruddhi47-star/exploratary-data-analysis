import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# 1. CREATE OUTPUT FOLDERS
# --------------------------------------------------

os.makedirs("output/graphs", exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

file_path = "data/student_performance.csv"

df = pd.read_csv(file_path)

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 3. BASIC INFORMATION
# --------------------------------------------------

print("\n1. FIRST 10 ROWS")
print("-" * 40)
print(df.head(10))

print("\n2. LAST 5 ROWS")
print("-" * 40)
print(df.tail())

print("\n3. DATASET SHAPE")
print("-" * 40)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n4. COLUMN NAMES")
print("-" * 40)
print(df.columns.tolist())

print("\n5. DATA TYPES")
print("-" * 40)
print(df.dtypes)

# --------------------------------------------------
# 4. DATASET INFORMATION
# --------------------------------------------------

print("\n6. DATASET INFORMATION")
print("-" * 40)
df.info()

# --------------------------------------------------
# 5. STATISTICAL SUMMARY
# --------------------------------------------------

print("\n7. STATISTICAL SUMMARY")
print("-" * 40)
print(df.describe())

# --------------------------------------------------
# 6. MISSING VALUES
# --------------------------------------------------

print("\n8. MISSING VALUES")
print("-" * 40)

missing_values = df.isnull().sum()

print(missing_values)

print("\nMissing percentage:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.round(2))

# --------------------------------------------------
# 7. DUPLICATE VALUES
# --------------------------------------------------

print("\n9. DUPLICATE ROWS")
print("-" * 40)

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)

# --------------------------------------------------
# 8. UNIQUE VALUES
# --------------------------------------------------

print("\n10. UNIQUE VALUES")
print("-" * 40)

for column in df.columns:
    print(column, ":", df[column].nunique())

# --------------------------------------------------
# 9. CATEGORICAL VARIABLES
# --------------------------------------------------

print("\n11. CATEGORICAL VARIABLES")
print("-" * 40)

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

print("Categorical columns:", list(categorical_columns))

for column in categorical_columns:
    print("\n", column)
    print(df[column].value_counts())

# --------------------------------------------------
# 10. NUMERICAL VARIABLES
# --------------------------------------------------

print("\n12. NUMERICAL VARIABLES")
print("-" * 40)

numerical_columns = df.select_dtypes(
    include=np.number
).columns

print("Numerical columns:", list(numerical_columns))

# --------------------------------------------------
# 11. MEAN, MEDIAN, MODE
# --------------------------------------------------

print("\n13. BASIC STATISTICS")
print("-" * 40)

for column in numerical_columns:

    print("\nColumn:", column)

    print("Mean:", df[column].mean())
    print("Median:", df[column].median())

    mode_value = df[column].mode()

    if len(mode_value) > 0:
        print("Mode:", mode_value.iloc[0])

    print("Standard Deviation:", df[column].std())

# --------------------------------------------------
# 12. CHECK NEGATIVE VALUES
# --------------------------------------------------

print("\n14. NEGATIVE VALUES")
print("-" * 40)

for column in numerical_columns:

    negative_count = (df[column] < 0).sum()

    print(column, ":", negative_count)

# --------------------------------------------------
# 13. OUTLIER DETECTION USING IQR
# --------------------------------------------------

print("\n15. OUTLIER DETECTION")
print("-" * 40)

outlier_summary = {}

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    outlier_summary[column] = len(outliers)

    print(column, ":", len(outliers), "outliers")

# --------------------------------------------------
# 14. CORRELATION ANALYSIS
# --------------------------------------------------

print("\n16. CORRELATION ANALYSIS")
print("-" * 40)

correlation = df[numerical_columns].corr()

print(correlation)

# --------------------------------------------------
# 15. SAVE CORRELATION HEATMAP
# --------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "output/graphs/correlation_heatmap.png"
)

plt.close()

# --------------------------------------------------
# 16. HISTOGRAMS
# --------------------------------------------------

print("\n17. CREATING HISTOGRAMS")

for column in numerical_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column].dropna(),
        kde=True
    )

    plt.title("Distribution of " + column)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    filename = (
        "output/graphs/histogram_"
        + column
        + ".png"
    )

    plt.savefig(filename)

    plt.close()

# --------------------------------------------------
# 17. BOXPLOTS
# --------------------------------------------------

print("\n18. CREATING BOXPLOTS")

for column in numerical_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df[column]
    )

    plt.title("Boxplot of " + column)

    plt.tight_layout()

    filename = (
        "output/graphs/boxplot_"
        + column
        + ".png"
    )

    plt.savefig(filename)

    plt.close()

# --------------------------------------------------
# 18. COUNT PLOTS FOR CATEGORICAL DATA
# --------------------------------------------------

print("\n19. CREATING CATEGORY PLOTS")

for column in categorical_columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=column
    )

    plt.title("Distribution of " + column)

    plt.xticks(rotation=45)

    plt.tight_layout()

    filename = (
        "output/graphs/countplot_"
        + column
        + ".png"
    )

    plt.savefig(filename)

    plt.close()

# --------------------------------------------------
# 19. SCATTER PLOTS
# --------------------------------------------------

print("\n20. CREATING SCATTER PLOTS")

if len(numerical_columns) >= 2:

    x_column = numerical_columns[0]
    y_column = numerical_columns[1]

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column
    )

    plt.title(
        x_column + " vs " + y_column
    )

    plt.tight_layout()

    plt.savefig(
        "output/graphs/scatter_plot.png"
    )

    plt.close()

# --------------------------------------------------
# 20. PAIR PLOT
# --------------------------------------------------

print("\n21. CREATING PAIR PLOT")

if len(numerical_columns) > 1:

    pair_plot = sns.pairplot(
        df[numerical_columns].dropna()
    )

    pair_plot.savefig(
        "output/graphs/pair_plot.png"
    )

    plt.close("all")

# --------------------------------------------------
# 21. DATA CLEANING
# --------------------------------------------------

print("\n22. CLEANING DATA")
print("-" * 40)

cleaned_df = df.copy()

# Remove duplicate rows
cleaned_df = cleaned_df.drop_duplicates()

# Fill numerical missing values with median
for column in numerical_columns:

    if cleaned_df[column].isnull().sum() > 0:

        cleaned_df[column] = cleaned_df[column].fillna(
            cleaned_df[column].median()
        )

# Fill categorical missing values with mode
for column in categorical_columns:

    if cleaned_df[column].isnull().sum() > 0:

        mode_value = cleaned_df[column].mode()

        if len(mode_value) > 0:

            cleaned_df[column] = cleaned_df[column].fillna(
                mode_value.iloc[0]
            )

# --------------------------------------------------
# 22. SAVE CLEANED DATASET
# --------------------------------------------------

cleaned_df.to_csv(
    "output/cleaned_student_data.csv",
    index=False
)

print(
    "Cleaned dataset saved successfully."
)

# --------------------------------------------------
# 23. FINAL DATA CHECK
# --------------------------------------------------

print("\n23. FINAL DATASET")
print("-" * 40)

print("Original rows:", len(df))
print("Cleaned rows:", len(cleaned_df))

print(
    "Remaining missing values:",
    cleaned_df.isnull().sum().sum()
)

print(
    "Remaining duplicate rows:",
    cleaned_df.duplicated().sum()
)

# --------------------------------------------------
# 24. FINAL MESSAGE
# --------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")
print("1. Cleaned dataset")
print("2. Histograms")
print("3. Boxplots")
print("4. Count plots")
print("5. Scatter plot")
print("6. Correlation heatmap")
print("7. Pair plot")

print("\nCheck the 'output' folder for results.")
