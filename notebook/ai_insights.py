import pandas as pd
from openai import OpenAI

# Load data
df = pd.read_csv("../data/business_data.csv")

# Aggregate
summary = df.groupby("Region").agg({
    "Revenue": "sum",
    "Utilization": "mean"
}).reset_index()

data_text = summary.to_string(index=False)

# AI
client = OpenAI(api_key="YOUR_API_KEY")

prompt = f"""
You are a business analyst.

Analyze the data and provide 3 key insights:

{data_text}
"""

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
