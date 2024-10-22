### README: Stock Price Prediction and Sentiment Analysis Using Machine Learning

---

## Project Overview

This project focuses on predicting stock prices by integrating machine learning models and sentiment analysis of financial news articles. The primary objective is to create a model that not only uses historical stock price data but also leverages the sentiment from the latest news articles to enhance prediction accuracy and provide valuable investment insights.

---

## Key Features

1. **News Scraping and Sentiment Analysis**: 
   - Web scraping of financial news headlines related to a specific stock ticker using `BeautifulSoup`.
   - Sentiment analysis of the news headlines using `TextBlob` to determine whether the sentiment is positive, negative, or neutral.
   
2. **Stock Data Collection**: 
   - The stock data is retrieved using the `yfinance` library for a given ticker symbol, covering historical data and key financial indicators.
   - Technical indicators like Moving Averages (SMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD) are calculated using the `ta` library.

3. **Stock Price Prediction**:
   - Linear Regression is applied to predict future stock prices based on historical data.
   - TPOT AutoML is used to automate model selection and hyperparameter tuning for time-series predictions.
   - The model is trained on past stock prices and technical indicators, and its performance is evaluated using error metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score.

4. **Visualization**:
   - Visual representation of sentiment analysis, including polarity and subjectivity scores.
   - Stock price trends, including moving averages and buy/sell signals based on technical indicators.
   - Prediction results for future stock prices compared with actual historical prices.

5. **Investment Suggestions**:
   - Based on sentiment analysis, the program offers a textual explanation of market sentiment, helping investors understand the underlying news and its potential impact on stock prices.

---

## Libraries Used

- `beautifulsoup4`: For web scraping news headlines.
- `lxml` and `html5lib`: HTML parsers used by `BeautifulSoup`.
- `textblob`: For sentiment analysis of news headlines.
- `yfinance`: For retrieving historical stock data.
- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical operations.
- `scikit-learn`: Machine learning library for Linear Regression, data splitting, and model evaluation.
- `tpot`: AutoML for automatic model selection and hyperparameter tuning.
- `ta`: Technical Analysis library for generating indicators like RSI and MACD.
- `matplotlib`: Data visualization.

---

## Installation Instructions

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   ```

2. **Install Required Libraries**:
   ```
   !pip install beautifulsoup4
   !pip install lxml
   !pip install html5lib
   !pip install textblob
   !pip install yfinance
   !pip install scikit-learn
   !pip install ta
   !pip install tpot
   ```

3. **Run the Jupyter Notebook or Colab**:
   - You can run the notebook in Google Colab or Jupyter Notebook.
   - Ensure to install all the required dependencies before running the script.

---

## Usage Instructions

1. **Input Ticker Symbol**:
   - Enter a valid stock ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`) when prompted to analyze the stock.

2. **View News Sentiment Analysis**:
   - The script scrapes the latest news headlines for the specified stock and performs sentiment analysis.
   - A bar chart is displayed showing sentiment polarity and subjectivity scores.

3. **Stock Price Predictions**:
   - After scraping news and calculating sentiment, the script fetches historical stock data and uses machine learning models to predict future stock prices.
   - Visualizations of stock price trends, moving averages, and future price predictions are provided.

4. **Investment Insights**:
   - Based on sentiment polarity, the program offers investment suggestions, such as whether to consider buying, holding, or being cautious with the stock.

---

## How It Works

1. **News Scraping**:
   - The function `scrape_news(ticker)` fetches news articles related to the stock from Google News using web scraping techniques.

2. **Sentiment Analysis**:
   - The function `analyze_sentiment(news_headlines)` performs sentiment analysis on the scraped news using `TextBlob`, which assigns a polarity score between -1 (negative) and 1 (positive), and a subjectivity score between 0 (objective) and 1 (subjective).

3. **Stock Data Retrieval**:
   - The `yfinance` library fetches stock price data for the ticker, including open, high, low, close prices, and volume.

4. **Stock Prediction**:
   - Two different approaches are used for stock price prediction:
     - **Linear Regression**: A traditional model to predict prices based on historical data.
     - **TPOT AutoML**: Automatically finds the best machine learning pipeline for predicting stock prices.
   
5. **Visualizing and Explaining Results**:
   - The program provides visual insights into the predictions through charts, along with textual explanations based on sentiment analysis.

---

## Example Results

- After running the script with the ticker `MSFT`:
   - Scraped news headlines show a positive sentiment, suggesting optimism in the market.
   - The model predicts a slight increase in stock prices, aligning with the positive news sentiment.

---

## Future Enhancements

- **Incorporate More Technical Indicators**: Adding more advanced technical indicators could improve the prediction accuracy.
- **Use of Deep Learning Models**: Implementing LSTM or GRU models for more accurate long-term stock price predictions.
- **News Impact Analysis**: Extending the project to analyze the impact of specific news categories (e.g., earnings, mergers) on stock price movements.

---

## License

This project is open-source and licensed under the MIT License.

---

## Author

**Subham Singha** - *Data Science Engineer*

--- 

Feel free to reach out if you have any questions or suggestions for improvement!
