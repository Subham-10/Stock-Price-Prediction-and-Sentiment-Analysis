# Install the necessary libraries in Colab
!pip install beautifulsoup4
!pip install lxml
!pip install html5lib
!pip install textblob
!pip install yfinance
!pip install scikit-learn

# Import required libraries
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Function to scrape news headlines and URLs for a given stock ticker
def scrape_news(ticker):
    # Google News URL for the specific ticker
    url = f'https://www.google.com/search?q={ticker}+stock+news&tbm=nws'

    # Request the webpage
    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching the news articles.")
        return []

    # Parse the webpage using BeautifulSoup with 'lxml' parser
    soup = BeautifulSoup(response.text, 'lxml')

    # Extract news headlines and their URLs
    headlines = []
    for item in soup.find_all('div', attrs={'class': 'BNeawe vvjwJb AP7Wnd'}):
        headline = item.get_text()
        link = item.find_parent('a')['href']  # Get the URL of the news article
        full_link = f"https://www.google.com{link}"
        headlines.append((headline, full_link))

    return headlines

# Function to analyze sentiment of the news headlines
def analyze_sentiment(news_headlines):
    analysis = TextBlob(' '.join(news_headlines))  # Join news headlines into a single string
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    return polarity, subjectivity

# Function to plot sentiment analysis results
def plot_sentiment(polarity, subjectivity):
    labels = ['Polarity', 'Subjectivity']
    values = [polarity, subjectivity]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['blue', 'green'])
    plt.title('Sentiment Analysis of News Headlines')
    plt.ylim(-1, 1)
    plt.ylabel('Score')
    plt.show()

# Define a function to provide investment suggestions
def investment_suggestion(polarity):
    if polarity > 0.1:
        return "Positive sentiment detected. Consider investing."
    elif polarity < -0.1:
        return "Negative sentiment detected. It might be risky to invest."
    else:
        return "Neutral sentiment detected. Further analysis is recommended before making an investment decision."

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# Prompt the user to enter the stock ticker
ticker = input("Enter the stock ticker symbol you want to analyze : ").upper()

# Scrape the news headlines and URLs for the given ticker
news_headlines = scrape_news(ticker)

if news_headlines:
    print(f"\nNews Headlines for {ticker}:\n")
    for i, (headline, link) in enumerate(news_headlines):
        print(f"{i + 1}. {headline}\n   Link: {link}")

    # Perform sentiment analysis on the news headlines
    only_headlines = [headline for headline, _ in news_headlines]
    polarity, subjectivity = analyze_sentiment(only_headlines)

    # Print sentiment analysis results
    print(f"\nSentiment Polarity: {polarity:.2f}")
    print(f"Sentiment Subjectivity: {subjectivity:.2f}")

    # Provide an investment suggestion based on sentiment
    suggestion = investment_suggestion(polarity)
    print(f"\nInvestment Suggestion: {suggestion}")

    # Plot polarity and subjectivity
    plot_sentiment(polarity, subjectivity)

    # Fetch and analyze stock data for the selected ticker
    current_date = datetime.now().strftime('%Y-%m-%d')
    data = fetch_stock_data(ticker, '2021-01-01', current_date)

    if not data.empty:
        # Predict future prices using Linear Regression
        data['Date'] = data.index
        data['Ordinal_Date'] = data['Date'].apply(lambda x: x.toordinal())

        X = data[['Ordinal_Date']].values
        y = data['Close'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\nMean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R-squared (R²): {r2:.2f}")

        # Future predictions
        future_dates = pd.date_range(start=current_date, end='2026-01-01')
        future_ordinals = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
        future_predictions = model.predict(future_ordinals)

        # Textual explanation and investment suggestions
        future_price = future_predictions[-1]
        current_price = data['Close'].iloc[-1]

        def explain_prediction(future_price, current_price, polarity):
            change_percent = ((future_price - current_price) / current_price) * 100
            explanation = f"The predicted stock price for {ticker} shows a {change_percent:.2f}% change from the current price of ${current_price:.2f}.\n"

            # Sentiment-based insights
            if polarity > 0.1:
                explanation += (
                    f"The news sentiment is positive with a polarity of {polarity:.2f}, indicating optimism about the future.\n"
                    f"This aligns well with the predicted price increase. Investors may view this as a good opportunity to invest in the stock."
                )
            elif polarity < -0.1:
                explanation += (
                    f"The news sentiment is negative with a polarity of {polarity:.2f}, indicating caution about the future.\n"
                    f"Despite the predicted price increase, investors might want to be cautious and wait for more stability."
                )
            else:
                explanation += (
                    f"The news sentiment is neutral with a polarity of {polarity:.2f}. The predicted stock price increase indicates moderate market confidence,\n"
                    f"but investors should do further research before making investment decisions."
                )

            return explanation

        print("\n" + explain_prediction(future_price, current_price, polarity))

    else:
        print("No data available for analysis.")
else:
    print(f"No news available for ticker {ticker}.")