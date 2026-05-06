# part-3-nlp-sequence-modeling-bitsom_ba_2511666
Part 3: NLP and Sequence Modeling Mini Project
About This Project
In this project, I worked with a customer support dataset to understand how Natural Language Processing (NLP) works. The main goal was to classify customer messages into three sentiments — positive, neutral, and negative — and learn how text is converted into numbers so that machine learning models can understand it.
Dataset Used
- File: customer_support_text_classification.csv
- Total records: 1,500 customer support messages
- Target: sentiment_label (positive, neutral, negative)
- The messages come from 5 channels: email, social media, phone, chat, and app
- Average message length: about 13 words
What I Did (Step by Step)
Task 1: Dataset Understanding
I loaded the dataset and explored it to understand its structure. I checked the number of records, the target classes, sample messages, average text length, and how the sentiments are distributed. The classes are fairly balanced — neutral (34.9%), negative (33.1%), and positive (31.9%). There were no missing values.
Task 2: Text Preprocessing
I cleaned the text data to make it ready for modeling. The steps I followed:
- Converted everything to lowercase
- Removed special characters, punctuation, and numbers (like ticket IDs)
- Removed extra spaces
- Split sentences into individual words (tokenization)
- Removed common stopwords like "the", "is", "a" that don't help with sentiment
After cleaning, the average word count dropped from 12.72 to 6.43 per message.
Task 3: Text Vectorization
I converted the cleaned text into numbers using three methods:
- Bag of Words (BoW) — counts how many times each word appears
- TF-IDF — gives higher scores to important words and lower scores to common ones
- Tokenizer-based Sequences — assigns a unique number to each word and pads all sequences to the same length (needed for deep learning models)
I also explained why we need to do this — because machine learning models only understand numbers, not raw text.
Task 4: Baseline Model
I built two simple models:
- Logistic Regression with TF-IDF features
- Naive Bayes with Bag of Words features
Both models achieved 100% accuracy because the dataset has very clear sentiment words (like "frustrated" for negative and "excellent" for positive). I evaluated them using accuracy, precision, recall, and F1-score.
Task 5: LSTM Sequence Model
I designed an LSTM (Long Short-Term Memory) model architecture and explained how it would process the text:
- Input Layer: padded word sequences (length 15)
- Embedding Layer: converts word IDs into 64-dimensional vectors
- LSTM Layer: reads words one by one and remembers context
- Dropout Layer: prevents overfitting
- Output Layer: predicts one of 3 sentiment classes using softmax
I also explained the loss function (categorical crossentropy) and optimizer (Adam) used.
Task 6: Attention and Transformer Reflection
I wrote about:
- Why RNNs struggle with long sentences (they forget earlier words)
- How LSTMs fix this using special gates that control memory
- How attention helps models focus on the most relevant words
- Why transformers are the foundation of modern AI tools like ChatGPT and BERT
Results
ModelAccuracyLogistic Regression + TF-IDF100%Naive Bayes + Bag of Words100%
ModelAccuracy
Model
Accuracy
Logistic Regression + TF-IDF100%
Logistic Regression + TF-IDF
100%
Naive Bayes + Bag of Words100%
Naive Bayes + Bag of Words
100%
