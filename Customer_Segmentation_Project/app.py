from flask import send_file
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv("data/ifood_df.csv")
df = df.dropna()

# Feature Engineering
df['TotalSpend'] = (
    df['MntWines'] +
    df['MntFruits'] +
    df['MntMeatProducts'] +
    df['MntFishProducts'] +
    df['MntSweetProducts'] +
    df['MntGoldProds']
)

features = df[['Income', 'TotalSpend', 'Recency']]

# Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(scaled_features)

# Generate chart
def generate_chart():
    plt.figure(figsize=(6, 4))
    plt.scatter(df['Income'], df['TotalSpend'], c=df['Segment'])
    plt.xlabel("Income")
    plt.ylabel("Total Spend")
    plt.title("Customer Segmentation")
    plt.tight_layout()
    plt.savefig("static/chart.png")
    plt.close()

generate_chart()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    income = float(request.form['income'])
    spend = float(request.form['spend'])
    recency = float(request.form['recency'])

    user_data = scaler.transform([[income, spend, recency]])
    segment = kmeans.predict(user_data)[0]

    segment_map = {
        0: "Premium Customers 💎",
        1: "At-Risk Customers ⚠️",
        2: "Loyal Customers ❤️"
    }

    return render_template(
        "index.html",
        result=segment_map.get(segment),
        chart=True
    )

@app.route('/download')
def download_chart():
    return send_file(
        "static/chart.png",
        as_attachment=True,
        download_name="customer_segmentation_chart.png"
    )


    return render_template(
        "index.html",
        result=segment_map.get(segment),
        chart=True
    )

if __name__ == "__main__":
    app.run(debug=True)
