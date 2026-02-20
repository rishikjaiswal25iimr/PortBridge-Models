# ============================================
# PORTBRIDGE AI-DRIVEN STRATEGIC DIAGNOSTICS
# NLP + CLUSTERING + VRIO (Clean Version)
# ============================================

!pip install scikit-learn --quiet

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================
# PART 1: NLP INDUSTRY THEME ANALYSIS
# ============================================

news = pd.read_csv("portbridge_industry_news_simulation.csv")

theme_counts = news["Theme_Label"].value_counts(normalize=True) * 100

print("\n========== INDUSTRY THEME DOMINANCE (%) ==========")
print(theme_counts.round(2))

plt.figure()
theme_counts.plot(kind='bar')
plt.title("Industry Theme Frequency (%)")
plt.ylabel("Percentage")
plt.show()


# ============================================
# PART 2: AI STRATEGIC GROUP MAPPING (K-Means)
# ============================================

df = pd.read_csv("portbridge_competitor_strategy_simulation.csv")

X = df[[
    "Digital_Maturity_Score",
    "Scale_Intensity_Score",
    "Pricing_Aggressiveness",
    "Operational_Integration"
]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Strategic_Group"] = kmeans.fit_predict(X_scaled)

print("\n========== STRATEGIC GROUP ASSIGNMENT ==========")
print(df[["Company", "Strategic_Group"]])

plt.figure()
plt.scatter(df["Digital_Maturity_Score"],
            df["Scale_Intensity_Score"],
            c=df["Strategic_Group"])

for i, txt in enumerate(df["Company"]):
    plt.annotate(txt,
                 (df["Digital_Maturity_Score"][i],
                  df["Scale_Intensity_Score"][i]))

plt.xlabel("Digital Maturity")
plt.ylabel("Scale Intensity")
plt.title("AI-Generated Strategic Group Map")
plt.show()


# ============================================
# PART 3: AI-AUGMENTED VRIO SCORING
# ============================================

df_vrio = pd.read_csv("portbridge_vrio_simulation.csv")

df_vrio["Sustainable_Advantage_Score"] = (
    df_vrio["Valuable"] *
    df_vrio["Rare"] *
    df_vrio["Inimitable"] *
    df_vrio["Organized"]
)

print("\n========== VRIO SCORING RESULTS ==========")
print(df_vrio)

plt.figure()
plt.bar(df_vrio["Resource"],
        df_vrio["Sustainable_Advantage_Score"])
plt.xticks(rotation=45)
plt.title("Sustainable Advantage Score by Resource")
plt.ylabel("Score (0 or 1)")
plt.show()
