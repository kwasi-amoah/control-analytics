#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# CONTINUOUS CONTROL MONITORING & ANOMALY DETECTION
# Financial Transaction Analysis — Internal Control Analytics
# Author: Paa Kwasi Amoah Apau-Danso
# =============================================================================
# PURPOSE:
# This script applies four automated control tests to financial transaction
# data, mimicking a Continuous Control Monitoring (CCM) framework used in
# internal control functions within financial services.
#
# CONTROL TESTS:
#   1. Benford's Law Analysis         — detects digit manipulation
#   2. Duplicate Payment Detection    — flags same amount/type/day combos
#   3. Round Number Analysis          — flags suspiciously round figures
#   4. After-Hours Transaction Check  — flags off-schedule postings
#
# OUTPUT:
# Each test exports a clean exceptions CSV to /outputs/ for Power BI ingestion
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# =============================================================================
# CONFIGURATION
# =============================================================================
 
DATA_PATH = "../data/transactions.csv"
OUTPUT_PATH = "../outputs/"
os.makedirs(OUTPUT_PATH, exist_ok=True)
 
# Business hours definition (for after-hours test)
BUSINESS_HOURS_START = 8   # 8:00 AM
BUSINESS_HOURS_END = 17    # 6:00 PM
 
# Round number threshold — flag amounts that are exact multiples of this
ROUND_NUMBER_THRESHOLD = 1000
 
# Benford deviation threshold — flag digits where deviation exceeds this %
BENFORD_DEVIATION_THRESHOLD = 5.0


# In[3]:


# =============================================================================
# SECTION 1: LOAD & PREPARE DATA
# =============================================================================
 
print("=" * 60)
print("LOADING TRANSACTION DATA")
print("=" * 60)
 
df = pd.read_csv(DATA_PATH)
 
print(f"✓ Dataset loaded: {len(df):,} transactions")
print(f"✓ Columns: {list(df.columns)}")
print(f"✓ Date range: Step {df['step'].min()} to Step {df['step'].max()}")
print(f"✓ Transaction types: {df['type'].unique()}")
print()


# In[4]:


# ── Data Cleaning & Feature Engineering ──────────────────────────────────────
 
# The PaySim dataset uses 'step' as a proxy for hours since simulation start.
# We convert this to a datetime for realistic analysis.
base_date = pd.Timestamp("2024-01-01")
df['transaction_datetime'] = base_date + pd.to_timedelta(df['step'], unit='h')
df['transaction_date'] = df['transaction_datetime'].dt.date
df['transaction_hour'] = df['transaction_datetime'].dt.hour
df['transaction_day'] = df['transaction_datetime'].dt.day_name()
 
# Standardise column references
# PaySim columns: step, type, amount, nameOrig, oldbalanceOrg,
#                 newbalanceOrig, nameDest, oldbalanceDest,
#                 newbalanceDest, isFraud, isFlaggedFraud
 
print("✓ Feature engineering complete")
print(f"✓ Amount range: ${df['amount'].min():,.2f} — ${df['amount'].max():,.2f}")
print()


# In[5]:


# =============================================================================
# SECTION 2: CONTROL TEST 1 — BENFORD'S LAW ANALYSIS
# =============================================================================
# WHY THIS MATTERS:
# In a genuine transaction population, the leading digit of amounts follows
# Benford's Law — 1 appears ~30% of the time, 9 only ~5%. When people
# fabricate or manipulate figures, they tend to use digits more uniformly,
# causing deviations. Internal auditors use this as a first-pass fraud screen.
# =============================================================================
 
print("=" * 60)
print("CONTROL TEST 1: BENFORD'S LAW ANALYSIS")
print("=" * 60)
 
# Expected Benford proportions for digits 1–9
benford_expected = {
    d: np.log10(1 + 1/d) * 100 for d in range(1, 10)
}
 
# Extract leading digit from transaction amounts (exclude zero/negative)
df_benford = df[df['amount'] > 0].copy()
df_benford['leading_digit'] = df_benford['amount'].astype(str).str[0].astype(int)
df_benford = df_benford[df_benford['leading_digit'].between(1, 9)]
 
# Calculate actual proportions
actual_counts = df_benford['leading_digit'].value_counts().sort_index()
actual_pct = (actual_counts / actual_counts.sum() * 100).round(2)
 
# Build comparison table
benford_df = pd.DataFrame({
    'digit': list(benford_expected.keys()),
    'expected_pct': [round(v, 2) for v in benford_expected.values()],
    'actual_pct': [actual_pct.get(d, 0) for d in range(1, 10)],
})
benford_df['deviation_pct'] = (
    benford_df['actual_pct'] - benford_df['expected_pct']
).round(2)
benford_df['flagged'] = benford_df['deviation_pct'].abs() > BENFORD_DEVIATION_THRESHOLD
benford_df['flag_label'] = benford_df['flagged'].map(
    {True: '⚠ EXCEPTION', False: '✓ Normal'}
)
 
print(benford_df.to_string(index=False))
print()
 
flagged_digits = benford_df[benford_df['flagged']]
if len(flagged_digits) > 0:
    print(f"⚠  {len(flagged_digits)} digit(s) exceed deviation threshold "
          f"({BENFORD_DEVIATION_THRESHOLD}%):")
    for _, row in flagged_digits.iterrows():
        print(f"   Digit {int(row['digit'])}: "
              f"Expected {row['expected_pct']}% | "
              f"Actual {row['actual_pct']}% | "
              f"Deviation {row['deviation_pct']:+.2f}%")
else:
    print("✓ No significant Benford deviations detected.")
 
# Export for Power BI
benford_df.to_csv(f"{OUTPUT_PATH}benford_analysis.csv", index=False)
print(f"\n✓ Exported: benford_analysis.csv")
 
# Plot Benford chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(1, 10)
width = 0.35
bars1 = ax.bar(x - width/2, benford_df['expected_pct'],
               width, label='Expected (Benford)', color='#2C6FBF', alpha=0.85)
bars2 = ax.bar(x + width/2, benford_df['actual_pct'],
               width, label='Actual', color='#E05C5C', alpha=0.85)
ax.set_xlabel('Leading Digit', fontsize=11)
ax.set_ylabel('Frequency (%)', fontsize=11)
ax.set_title("Benford's Law: Expected vs Actual Leading Digit Distribution",
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}benford_chart.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: benford_chart.png")
print()


# In[6]:


# =============================================================================
# SECTION 3: CONTROL TEST 2 — DUPLICATE PAYMENT DETECTION
# =============================================================================
# WHY THIS MATTERS:
# Duplicate payments are a common control failure — same vendor, same amount,
# same day. They can be accidental (system errors) or deliberate (fraud).
# An automated control flags these for human review before payment clears.
# =============================================================================
 
print("=" * 60)
print("CONTROL TEST 2: DUPLICATE PAYMENT DETECTION")
print("=" * 60)
 
# Flag transactions with identical: destination account + amount + date + type
duplicate_mask = df.duplicated(
    subset=['nameDest', 'amount', 'transaction_date', 'type'],
    keep=False
)
df_duplicates = df[duplicate_mask].copy()
df_duplicates['exception_type'] = 'Duplicate Payment'
df_duplicates['exception_detail'] = (
    'Same destination, amount, type, and date as another transaction'
)
 
# Summarise
dup_groups = df_duplicates.groupby(
    ['nameDest', 'amount', 'transaction_date', 'type']
).size().reset_index(name='occurrence_count')
 
print(f"⚠  {len(df_duplicates):,} transactions flagged as potential duplicates")
print(f"   Representing {len(dup_groups):,} unique duplicate groups")
print(f"   Total value at risk: ${df_duplicates['amount'].sum():,.2f}")
print()
 
# Show top 10 by amount
top_dups = dup_groups.sort_values('amount', ascending=False).head(10)
print("Top 10 duplicate groups by amount:")
print(top_dups.to_string(index=False))
 
# Export
df_duplicates[['step', 'type', 'amount', 'nameOrig', 'nameDest',
               'transaction_datetime', 'exception_type',
               'exception_detail']].to_csv(
    f"{OUTPUT_PATH}duplicate_exceptions.csv", index=False
)
print(f"\n✓ Exported: duplicate_exceptions.csv")
print()


# In[7]:


# =============================================================================
# SECTION 4: CONTROL TEST 3 — ROUND NUMBER ANALYSIS
# =============================================================================
# WHY THIS MATTERS:
# Fraudsters and people manipulating entries tend to use round numbers
# (e.g. exactly $5,000 or $10,000) because they're easier to remember and
# calculate. A spike in perfectly round amounts is a red flag for auditors.
# =============================================================================
 
print("=" * 60)
print("CONTROL TEST 3: ROUND NUMBER ANALYSIS")
print("=" * 60)
 
# Flag amounts that are exact multiples of the threshold
df['is_round'] = (df['amount'] % ROUND_NUMBER_THRESHOLD == 0) & (df['amount'] > 0)
df_round = df[df['is_round']].copy()
df_round['exception_type'] = 'Round Number'
df_round['exception_detail'] = (
    f'Amount is exact multiple of {ROUND_NUMBER_THRESHOLD:,}'
)
 
round_rate = len(df_round) / len(df) * 100
 
print(f"⚠  {len(df_round):,} transactions flagged as round numbers "
      f"({round_rate:.2f}% of total)")
print(f"   Total value: ${df_round['amount'].sum():,.2f}")
print()
 
# Distribution of round amounts
round_dist = (df_round['amount']
              .value_counts()
              .sort_index()
              .head(15)
              .reset_index())
round_dist.columns = ['amount', 'count']
print("Most frequent round amounts:")
print(round_dist.to_string(index=False))
 
# Export
df_round[['step', 'type', 'amount', 'nameOrig', 'nameDest',
          'transaction_datetime', 'exception_type',
          'exception_detail']].to_csv(
    f"{OUTPUT_PATH}round_number_exceptions.csv", index=False
)
print(f"\n✓ Exported: round_number_exceptions.csv")
print()
 


# In[8]:


# =============================================================================
# SECTION 5: CONTROL TEST 4 — AFTER-HOURS TRANSACTION DETECTION
# =============================================================================
# WHY THIS MATTERS:
# Transactions posted outside normal business hours warrant additional scrutiny.
# This is especially relevant for internal journal entries and transfers where
# legitimate activity should follow normal working patterns. Unusual timing
# can indicate unauthorised access or deliberate timing to avoid oversight.
# =============================================================================
 
print("=" * 60)
print("CONTROL TEST 4: AFTER-HOURS TRANSACTION DETECTION")
print("=" * 60)
 
df_after_hours = df[
    (df['transaction_hour'] < BUSINESS_HOURS_START) |
    (df['transaction_hour'] >= BUSINESS_HOURS_END)
].copy()
df_after_hours['exception_type'] = 'After-Hours Transaction'
df_after_hours['exception_detail'] = df_after_hours['transaction_hour'].apply(
    lambda h: f'Posted at {h:02d}:00 — outside business hours '
              f'({BUSINESS_HOURS_START:02d}:00–{BUSINESS_HOURS_END:02d}:00)'
)
 
after_hours_rate = len(df_after_hours) / len(df) * 100
 
print(f"⚠  {len(df_after_hours):,} after-hours transactions flagged "
      f"({after_hours_rate:.2f}% of total)")
 
# Hourly distribution
hourly = df_after_hours['transaction_hour'].value_counts().sort_index()
print("\nAfter-hours transaction volume by hour:")
for hour, count in hourly.items():
    bar = '█' * (count // max(hourly.max() // 30, 1))
    print(f"  {hour:02d}:00  {bar} {count:,}")
 
# Export
df_after_hours[['step', 'type', 'amount', 'nameOrig', 'nameDest',
                'transaction_datetime', 'transaction_hour',
                'exception_type', 'exception_detail']].to_csv(
    f"{OUTPUT_PATH}after_hours_exceptions.csv", index=False
)
print(f"\n✓ Exported: after_hours_exceptions.csv")
print()


# In[9]:


# =============================================================================
# SECTION 6: MASTER EXCEPTIONS SUMMARY
# =============================================================================
# Combine all exceptions into one consolidated table for Power BI overview page
# =============================================================================
 
print("=" * 60)
print("BUILDING MASTER EXCEPTIONS SUMMARY")
print("=" * 60)
 
common_cols = ['step', 'type', 'amount', 'nameOrig', 'nameDest',
               'transaction_datetime', 'exception_type', 'exception_detail']
 
master = pd.concat([
    df_duplicates[common_cols],
    df_round[common_cols],
    df_after_hours[common_cols]
], ignore_index=True)
 
master.to_csv(f"{OUTPUT_PATH}master_exceptions.csv", index=False)
 
# Summary stats
summary = master.groupby('exception_type').agg(
    exception_count=('exception_type', 'count'),
    total_value=('amount', 'sum'),
    avg_value=('amount', 'mean')
).reset_index()
summary['total_value'] = summary['total_value'].round(2)
summary['avg_value'] = summary['avg_value'].round(2)
summary['exception_rate_pct'] = (
    summary['exception_count'] / len(df) * 100
).round(3)
 
print("\nEXCEPTION SUMMARY TABLE")
print("-" * 70)
print(summary.to_string(index=False))
print("-" * 70)
print(f"\nTotal exceptions flagged: {len(master):,}")
print(f"Total transaction population: {len(df):,}")
print(f"Overall exception rate: {len(master)/len(df)*100:.2f}%")
print(f"Total value under review: ${master['amount'].sum():,.2f}")
 
summary.to_csv(f"{OUTPUT_PATH}exceptions_summary.csv", index=False)
print(f"\n✓ Exported: master_exceptions.csv")
print(f"✓ Exported: exceptions_summary.csv")
print()
 
# Also export full dataset with exception flags for Power BI drill-through
df['is_duplicate'] = duplicate_mask
df['is_round_number'] = df['is_round']
df['is_after_hours'] = (
    (df['transaction_hour'] < BUSINESS_HOURS_START) |
    (df['transaction_hour'] >= BUSINESS_HOURS_END)
)
df['exception_count'] = (
    df['is_duplicate'].astype(int) +
    df['is_round_number'].astype(int) +
    df['is_after_hours'].astype(int)
)
df['has_exception'] = df['exception_count'] > 0
 
# Export a sample for Power BI (full dataset may be large)
df_sample = df.sample(n=min(100000, len(df)), random_state=42)
df_sample.to_csv(f"{OUTPUT_PATH}transactions_flagged.csv", index=False)
print(f"✓ Exported: transactions_flagged.csv (sample for Power BI)")
 
print()
print("=" * 60)
print("ALL CONTROL TESTS COMPLETE")
print(f"All outputs saved to: {OUTPUT_PATH}")
print("=" * 60)


# In[ ]:




