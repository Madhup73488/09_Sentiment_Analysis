# EDA packages
import matplotlib
import pandas as pd

# Data viz pkg
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv("covid19_tweets.csv")

# preview
print(df.head())

"""
Text
    -> Text Preprocessing
    -> Sentiment Analysis
    -> Keyword Extraction
    -> Entity Extraction
"""

# Checking Columns
print(df.columns)

# Datatype
print(df.dtypes)

# Source/ Value Count/Distribution of the Sources
print(df['source'].unique())
print(df['source'].value_counts())

# printing the top 20 value_counts
print(df['source'].value_counts().nlargest(20))

# plotting the top 20 value_counts
plt.figure(figsize=(8, 6))  # Set the figure size (optional)
df['source'].value_counts().nlargest(20).plot(kind='bar', color='skyblue')
# Plot a bar chart
plt.title('Value Counts')  # Set the title
plt.xlabel('Sources')  # Set the label for x-axis
plt.ylabel('Count')  # Set the label for y-axis
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for y-axis
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

# Text Analysis of tweet
# Load Text Cleaing Package
import neattext.functions as nfx

# Methods/Attrib
print(dir(nfx))

print(df['text'].iloc[2])

"""
    -> Remove mentions/userhandles
    -> remove hashtags
    -> url's
    -> emoji's
    -> special char
"""
print(df.head())
df['text'].apply(nfx.extract_hashtags)
df['extracted_hashtags'] = df['text'].apply(nfx.extract_hashtags)
print(df[['extracted_hashtags','hashtags']])

# Cleaning Text

# Removing hashtags
df['clean_tweet'] = df['text'].apply(nfx.remove_hashtags)
print(df[['text', 'clean_tweet']])

# Removing user handles
df['clean_tweet']=df['clean_tweet'].apply(lambda x :nfx.remove_userhandles(x))
print(df[['text','clean_tweet']])

print(df['clean_tweet'].iloc[10])

# Removing multiple spaces
df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_multiple_spaces)
print(df['clean_tweet'].iloc[10])

# Removing urls
df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_urls)

print(df['clean_tweet'].iloc[10])

# Removing punctuations
df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_puncts)

# Comparing real tweets with cleaned tweets
print(df[['text', 'clean_tweet']])


# Sentiment Analysis
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity > 0 :
        sentiment_label = 'Positive'
    elif sentiment_polarity < 0:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    result = {'polarity':sentiment_polarity,
              'subjectivity':sentiment_subjectivity,
              'sentiment':sentiment_label}
    return result

# Checking the sentiment of an example text
ex1 = df['clean_tweet'].iloc[0]
print(get_sentiment(ex1))

# applying sentiment analysis to entire tweets
df['sentiment_results'] = df['clean_tweet'].apply(get_sentiment)

print(df['sentiment_results'])

print(df['sentiment_results'].iloc[0]['sentiment'])


# Extracting 'text' and 'sentiment_results' columns
selected_columns = df[['clean_tweet', 'sentiment_results']]

# Assuming 'sentiment_results' contains dictionary-like objects, extract 'sentiment' value
selected_columns['sentiment_value'] = selected_columns['sentiment_results'].\
    apply(lambda x: x['sentiment'])

# Dropping 'sentiment_results' column if not needed in the final CSV
selected_columns = selected_columns.drop(columns=['sentiment_results'])

# Save the extracted data to a new CSV file
selected_columns.to_csv('extracted_data.csv', index=False)
