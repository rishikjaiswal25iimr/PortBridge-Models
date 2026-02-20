# ============================================
# PORTBRIDGE EXECUTIVE AI CONTROL TOWER
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1. Load Data
# ------------------------------
freight = pd.read_csv("portbridge_freight_simulation.csv")
credit = pd.read_csv("portbridge_credit_risk_simulation.csv")
customs = pd.read_csv("portbridge_customs_nlp_simulation.csv")

# ------------------------------
# 2. KPI 1 – Freight Volatility
# ------------------------------

avg_rate = freight["SCFI_Index"].mean()
volatility = freight["SCFI_Index"].std()

print("\n========== EXECUTIVE DASHBOARD ==========")
print("Average Freight Index:", round(avg_rate,2))
print("Freight Volatility (Std Dev):", round(volatility,2))

plt.figure()
plt.plot(freight["Date"][:100], freight["SCFI_Index"][:100])
plt.xticks(rotation=45)
plt.title("Freight Rate Trend (Sample)")
plt.show()

# ------------------------------
# 3. KPI 2 – Credit Risk Exposure
# ------------------------------

total_clients = len(credit)
default_rate = credit["Default_Flag"].mean() * 100

print("\nTotal SME Clients:", total_clients)
print("Portfolio Default Rate:", round(default_rate,2), "%")

plt.figure()
plt.hist(credit["DSO_Variance"], bins=20)
plt.title("DSO Variance Distribution (Risk Indicator)")
plt.show()

# ------------------------------
# 4. KPI 3 – Customs Automation
# ------------------------------

total_invoices = len(customs)
automation_target = int(total_invoices * 0.4)

print("\nTotal Invoices:", total_invoices)
print("Target AI Automation Volume:", automation_target)

plt.figure()
plt.pie(
    [automation_target, total_invoices - automation_target],
    labels=["AI Processed","Manual Processed"],
    autopct='%1.1f%%'
)
plt.title("Customs Automation Mix")
plt.show()

# ------------------------------
# 5. Combined Strategic Risk Score
# ------------------------------

# Simple strategic composite indicator (for executive reporting)
risk_score = (
    (volatility / 1000) * 0.4 +
    (default_rate / 100) * 0.4 +
    (1 - 0.4) * 0.2
)

print("\nComposite Strategic Risk Index:", round(risk_score,3))
