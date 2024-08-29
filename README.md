# Sentiment Analysis and Modeling for Amazon Reviews

## Overview

This project focuses on sentiment analysis and sentiment modeling using Amazon product reviews. The analysis involves preprocessing the text data, visualizing word frequencies, and creating sentiment models using various machine learning techniques. The end goal is to predict the sentiment (positive or negative) of reviews based on the text.

## Project Structure

The project consists of five main stages:

1. **Text Preprocessing**: Preparing the text data for analysis by cleaning, normalizing, and tokenizing the text.
2. **Text Visualization**: Visualizing the text data using term frequency counts and word clouds.
3. **Sentiment Analysis**: Applying sentiment analysis techniques to classify reviews as positive or negative.
4. **Feature Engineering**: Creating features from the text data to be used in modeling.
5. **Sentiment Modeling**: Building and evaluating models to predict review sentiments.

## Requirements

To run this project, you need to have Python and the following libraries installed:

- `nltk`
- `textblob`
- `wordcloud`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `Pillow`

You can install the required packages using the following commands:

```bash
pip install nltk
pip install textblob
pip install wordcloud
```

## Dataset

The project uses an Amazon reviews dataset in CSV format (`amazon_reviews.csv`). Make sure this file is located in the `datasets` folder within your project directory.

## Project Steps

### 1. Text Preprocessing

- **Lowercasing**: Converts all text to lowercase to ensure uniformity.
- **Removing Punctuation**: Strips out punctuation marks using regular expressions.
- **Removing Numbers**: Removes digits to clean up the text further.
- **Removing Stopwords**: Filters out common English stopwords using the NLTK library.
- **Removing Rare Words**: Identifies and removes words that appear only once in the dataset.
- **Tokenization**: Splits the text into individual words.
- **Lemmatization**: Reduces words to their base forms using the WordNet lemmatizer.

### 2. Text Visualization

- **Term Frequency Calculation**: Counts the frequency of each word in the dataset.
- **Bar Plot**: Visualizes the most frequent words using bar plots.
- **Word Cloud**: Generates word clouds to show the most common words visually. Includes options for customizing the appearance using masks.

### 3. Sentiment Analysis

- Uses the `SentimentIntensityAnalyzer` from NLTK to compute sentiment scores of the reviews.
- Sentiment scores range from -1 (negative) to +1 (positive).

### 4. Feature Engineering

- Creates a binary sentiment label (`pos` for positive, `neg` for negative) based on sentiment scores.
- Encodes sentiment labels numerically for model training.
- Extracts various features using count vectors, TF-IDF vectors, and n-grams.

### 5. Sentiment Modeling

Two machine learning models are used to predict the sentiment of the reviews:

1. **Logistic Regression**:
   - Trains a logistic regression model using TF-IDF word-level features.
   - Evaluates model performance using 5-fold cross-validation.
   
2. **Random Forest**:
   - Trains a random forest model using count vectors, TF-IDF word-level, and TF-IDF n-gram level features.
   - Optimizes hyperparameters using GridSearchCV.

## Results

- The project evaluates models using cross-validation accuracy scores to compare their performance.
- The final model's performance is further refined using hyperparameter tuning.

## How to Run the Project

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies listed above.
3. Ensure the `amazon_reviews.csv` dataset is in the `datasets` folder.
4. Run the code cells in the script step-by-step or execute the entire script to see the complete workflow.

## Conclusion

This project demonstrates the process of building a sentiment analysis pipeline, from preprocessing and feature extraction to model training and evaluation. The insights gained can be applied to other sentiment analysis tasks or adapted for different types of text data.
