# ============================================
# PORTBRIDGE CUSTOMS NLP AUTOMATION
# Basic + Advanced + Accuracy Comparison
# ============================================

# Install required library
!pip install spacy --quiet

import pandas as pd
import re
import spacy
import numpy as np

# Load small English model
!python -m spacy download en_core_web_sm --quiet
nlp = spacy.load("en_core_web_sm")

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
df = pd.read_csv("portbridge_customs_nlp_simulation.csv")

print("Dataset Preview:")
print(df[['Invoice_No','Raw_Text']].head())

# --------------------------------------------
# PART 1: BASIC EXTRACTION (REGEX BASED)
# --------------------------------------------

def regex_extraction(text):
    hs = re.search(r'HS Code:\s*(\d+)', text)
    weight = re.search(r'Gross Weight:\s*([\d.]+)', text)
    value = re.search(r'Invoice Value:\s*INR\s*([\d.]+)', text)

    return {
        "HS_Code_Extracted": hs.group(1) if hs else None,
        "Weight_Extracted": float(weight.group(1)) if weight else None,
        "Value_Extracted": float(value.group(1)) if value else None
    }

regex_results = df['Raw_Text'].apply(regex_extraction)
regex_df = pd.json_normalize(regex_results)

df_regex = pd.concat([df, regex_df], axis=1)

print("\nRegex Extraction Sample:")
print(df_regex[['HS_Code','HS_Code_Extracted']].head())

# --------------------------------------------
# PART 2: ADVANCED EXTRACTION (spaCy NER)
# --------------------------------------------

def spacy_extraction(text):
    doc = nlp(text)
    entities = {
        "ORG": [],
        "MONEY": [],
        "QUANTITY": []
    }

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)

    return entities

df['NER_Output'] = df['Raw_Text'].apply(spacy_extraction)

print("\nSample NER Output:")
print(df[['Invoice_No','NER_Output']].head())

# --------------------------------------------
# PART 3: ACCURACY COMPARISON
# --------------------------------------------

# Compare Regex Extracted vs True Values

hs_accuracy = np.mean(df_regex['HS_Code'].astype(str) == df_regex['HS_Code_Extracted'].astype(str))
weight_accuracy = np.mean(np.isclose(df_regex['Gross_Weight_KG'], df_regex['Weight_Extracted'], atol=0.1))
value_accuracy = np.mean(np.isclose(df_regex['Invoice_Value_INR'], df_regex['Value_Extracted'], atol=1))

print("\n========== EXTRACTION ACCURACY ==========")
print("HS Code Accuracy: ", round(hs_accuracy*100,2), "%")
print("Weight Accuracy: ", round(weight_accuracy*100,2), "%")
print("Invoice Value Accuracy: ", round(value_accuracy*100,2), "%")

overall_accuracy = np.mean([hs_accuracy, weight_accuracy, value_accuracy])
print("Overall Extraction Accuracy:", round(overall_accuracy*100,2), "%")

# --------------------------------------------
# AUTOMATION KPI (For Dashboard)
# --------------------------------------------

automation_rate = overall_accuracy * 100
print("\nAutomation Readiness Score:", round(automation_rate,2), "%")

# ============================================
# PORTBRIDGE KPI DASHBOARD (AI IMPACT METRICS)
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# 1. Basic KPI Calculations
# --------------------------------------------

total_invoices = len(df)

# Using previously calculated automation rate
ai_accuracy = overall_accuracy  # from previous code
automation_rate = ai_accuracy

# Assume automation applied to low-complexity invoices only (realistic)
ai_processed = int(total_invoices * 0.4)   # 40% automation target
manual_processed = total_invoices - ai_processed

# Time assumption (business simulation)
manual_time_per_invoice = 15  # minutes
ai_time_per_invoice = 3       # minutes

total_manual_time = total_invoices * manual_time_per_invoice
total_ai_time = (ai_processed * ai_time_per_invoice) + (manual_processed * manual_time_per_invoice)

time_saved = total_manual_time - total_ai_time

# Cost assumption (₹500 per manual processing hour)
cost_per_hour = 500
cost_saved = (time_saved / 60) * cost_per_hour

# --------------------------------------------
# 2. Print KPI Summary
# --------------------------------------------

print("\n========== PORTBRIDGE AI KPI DASHBOARD ==========")
print("Total Invoices:", total_invoices)
print("Automation Accuracy:", round(automation_rate*100,2), "%")
print("Invoices Processed by AI:", ai_processed)
print("Invoices Processed Manually:", manual_processed)
print("Total Time Saved (minutes):", round(time_saved,2))
print("Estimated Cost Saved (INR):", round(cost_saved,2))

# --------------------------------------------
# 3. Visualization 1: AI vs Manual Split
# --------------------------------------------

plt.figure()
plt.pie(
    [ai_processed, manual_processed],
    labels=["AI Processed", "Manual Processed"],
    autopct='%1.1f%%'
)
plt.title("AI vs Manual Invoice Processing Split")
plt.show()

# --------------------------------------------
# 4. Visualization 2: Time Comparison
# --------------------------------------------

plt.figure()
plt.bar(
    ["Before AI (Manual Only)", "After AI Integration"],
    [total_manual_time, total_ai_time]
)
plt.title("Processing Time Comparison (Minutes)")
plt.ylabel("Total Processing Time")
plt.show()

# --------------------------------------------
# 5. Visualization 3: Cost Savings Indicator
# --------------------------------------------

plt.figure()
plt.bar(["Estimated Cost Saved (INR)"], [cost_saved])
plt.title("Estimated Administrative Cost Savings")
plt.show()
