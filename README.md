# customer-segmentation-flask
A machine learning–based customer segmentation web application built with Flask. The project uses K-Means clustering to group customers based on income, spending behavior, and recency. It includes interactive visualizations, real-time predictions, and a downloadable segmentation chart for business insights.# Customer Segmentation Web Application 🚀

A machine learning–based **Customer Segmentation System** built using **Python, Flask, and K-Means Clustering**.  
This project analyzes customer purchasing behavior and groups customers into meaningful segments to support **data-driven marketing and business decisions**.

---

## 📌 Project Overview

Customer segmentation is a critical task in business analytics.  
This project uses **unsupervised machine learning (K-Means clustering)** to segment customers based on:

- Income
- Total spending behavior
- Purchase recency

The trained model is deployed as a **Flask web application** that allows users to:
- Enter customer details
- Predict customer segment in real time
- Visualize customer clusters
- Download segmentation charts

---

## 🎯 Objectives

- Perform customer segmentation using machine learning
- Apply data preprocessing and feature scaling
- Visualize customer clusters
- Build an interactive web interface using Flask
- Enable chart download for reporting and analysis

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Framework:** Flask  
- **Machine Learning:** Scikit-learn (K-Means Clustering)  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **IDE:** VS Code  

---

## 📂 Project Structure

customer-segmentation-flask/
│
├── data/
│ └── ifood_df.csv
│
├── static/
│ ├── style.css
│ └── chart.png
│
├── templates/
│ └── index.html
│
├── app.py
├── requirements.txt
└── README.md

---

## 📊 Dataset Description

The dataset contains customer demographic and purchasing information, including:
- Income
- Spending on different product categories
- Recency of last purchase

A new feature **TotalSpend** is created by combining multiple spending attributes.

---

## ⚙️ How It Works

1. Load and clean customer data  
2. Perform feature engineering  
3. Apply feature scaling using `StandardScaler`  
4. Use **Elbow Method** to determine optimal clusters  
5. Apply **K-Means clustering**  
6. Visualize clusters using scatter plots  
7. Deploy results through a Flask web interface  

---

## ▶️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/customer-segmentation-flask.git
cd customer-segmentation-flask
