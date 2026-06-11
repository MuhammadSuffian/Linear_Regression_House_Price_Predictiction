# House Price Prediction with AI

A full-stack House Price Prediction web application built with Streamlit, powered by a multi-variable Linear Regression model and enhanced with an LLM for conversational price insights.

## Overview

This project predicts house prices using a machine learning model trained on a real-world housing dataset from Kaggle. To make predictions more intuitive and user-friendly, the app integrates the **openai/gpt-oss-120b** model, which explains predictions in natural language.

The project demonstrates how academic machine learning concepts can be transformed into practical, production-ready applications.

## Features

* Multi-variable Linear Regression for house price prediction
* Conversational AI-powered explanations using LLM
* Interactive Streamlit web interface
* Real-time predictions based on user inputs
* Clean and user-friendly design
* End-to-end deployment

## Tech Stack

* **Machine Learning:** Linear Regression (Scikit-learn)
* **LLM:** openai/gpt-oss-120b
* **Frontend & Backend:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## Dataset

Housing dataset sourced from Kaggle:

* House features such as area, bedrooms, bathrooms, stories, parking, and more
* Target variable: House Price

## Project Structure

```text
├── app.py
├── model.pkl
├── housing.csv
├── notebooks/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repository-url>
cd house-price-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Live Demo

Live Application: [https://lnkd.in/dZfMRy74](https://muhammadsuffian-linear-regressi-house-price-app-with-llm-loohfa.streamlit.app/)

## Key Takeaway

This project started as a coursework machine learning model and evolved into a complete AI-powered web application. It highlights how classroom projects can be transformed into real-world solutions with a little creativity and engineering effort.

## Author

**Muhammad Suffian**
