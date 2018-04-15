# Automated Machine Learning to Identify SEO-Spam Websites and Illegal Websites

Requirements:
Numpy
Pandas
Pylab
scikit
Jupyter


Objective: 

Identify the spam websites and pirated movies websites using certain techniques to extract features from text. sklearn.feature_extraction.text.CountVectorizer extracts features based on word count. Then deploying the feature-vectors such as to Multinomial Naive bayes Classifier to classify Spam website/ Non Spam websites. sklearn.feature_extraction.text.TfidfVectorizer extracts features based on word count giving less weightage to frequent words and more weigtage to rare words.

Note: 
Training a large data set takes very longer time, due to time constrains. I have prepared dataset limited around 150 count. This is for demonstration purpose, Hence I have used less data set to test it and it works sucessfully! :D 
