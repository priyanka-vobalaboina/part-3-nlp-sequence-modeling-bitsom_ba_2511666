# part-3-nlp-sequence-modeling-bitsom_ba_2511666

============================================================
TASK 3: TEXT VECTORIZATION
============================================================

WHY TEXT MUST BE CONVERTED INTO VECTORS:
-----------------------------------------
Machine learning models are mathematical algorithms — they can only
perform calculations on numbers, not on raw text or words.
For example, a model cannot directly understand the sentence:
"The refund process was fast and convenient"
We need to convert it into a numerical format (vector) like:
[0.43, 0.37, 0.29, 0.45, 0.0, 0.0, ...]
Reasons why vectorization is necessary:
1. Mathematical Operations: Models use operations like multiplication,addition, and dot products. These only work with numbers.
2. Pattern Recognition: Algorithms find patterns by comparing numerical distances between data points. Text has no inherent distance measure.
3. Fixed Input Size: Most models require inputs of a fixed size.Vectorization converts variable-length text into fixed-size arrays.
4. Capturing Meaning: Good vectorization methods (like TF-IDF) can capture which words are more important, helping the model learn which words indicate positive, negative, or neutral sentiment.
Without vectorization, no machine learning model — whether simple
(Logistic Regression) or complex (Neural Networks) — can process text.

============================================================
TASK 5: SEQUENCE MODEL - LSTM ARCHITECTURE DESIGN
============================================================
Since full model training requires TensorFlow/GPU resources, I am presenting
a clear architecture design with explanation of how the model processes input.

==============================================================
LSTM MODEL ARCHITECTURE FOR SENTIMENT CLASSIFICATION
==============================================================

1. INPUT SEQUENCE:
------------------
   - Each customer message is converted to a sequence of word indices
   - All sequences are padded/truncated to a fixed length of 15
   - Shape: (batch_size, 15)
   - Example input: [11, 47, 48, 49, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   - Here, each number represents a word and 0s are padding

2. EMBEDDING LAYER:
-------------------
   - Purpose: Converts each word index into a meaningful dense vector
   - Vocabulary size: 148 unique words
   - Embedding dimension: 64
   - Output shape: (batch_size, 15, 64)
   - Each word becomes a 64-dimensional vector that captures its meaning
   - These vectors are learned during training

3. RECURRENT/SEQUENCE LAYER (LSTM):
------------------------------------
   - Type: LSTM (Long Short-Term Memory)
   - Units: 64
   - How it works: Reads the sequence one word at a time from left to right
   - At each step, it updates its hidden state and cell state
   - The cell state acts as a memory that carries important information
   - Output shape: (batch_size, 64) — returns the final hidden state
   - This final state captures the meaning of the entire sentence
   - Dropout (0.3) is added after LSTM to prevent overfitting

4. OUTPUT LAYER:
----------------
   - Type: Dense (fully connected) layer
   - Units: 3 (one for each sentiment class: positive, neutral, negative)
   - Activation: softmax (converts raw scores into probabilities)
   - Output shape: (batch_size, 3)
   - Example output: [0.05, 0.03, 0.92] means 92% chance of positive

5. LOSS FUNCTION:
-----------------
   - categorical_crossentropy
   - This is the standard loss function for multi-class classification
   - It measures how far the predicted probabilities are from the true labels
   - The model tries to minimize this loss during training

6. EVALUATION METRIC:
---------------------
   - Primary: Accuracy (percentage of correct predictions)
   - Additional: Precision, Recall, F1-Score per class
   - These help us understand performance for each sentiment separately

============================================================
TASK 6: ATTENTION AND TRANSFORMER REFLECTION
============================================================

--------------------------------------------------------------
1. WHY RNNs STRUGGLE WITH LONG-TERM DEPENDENCIES
--------------------------------------------------------------
RNNs process text one word at a time, passing information forward through
a hidden state. The problem is that as sequences get longer, information
from earlier words gets "diluted" or lost. This is called the vanishing
gradient problem - during training, gradients become extremely small as
they are multiplied through many time steps, making it hard for the model
to learn connections between distant words.

Example: In "The movie that I watched last weekend with my friends at the
new theater downtown was absolutely terrible", an RNN struggles to connect
"movie" with "terrible" because they are far apart.

--------------------------------------------------------------
2. HOW LSTMs HELP WITH MEMORY
--------------------------------------------------------------
LSTMs (Long Short-Term Memory) solve this by introducing a "cell state" -
think of it as a conveyor belt that carries information across time steps.
LSTMs have three gates:

- Forget Gate: Decides what old information to throw away
- Input Gate: Decides what new information to store
- Output Gate: Decides what information to output

These gates allow the model to selectively remember or forget information,
making it much better at capturing long-range dependencies than vanilla RNNs.

--------------------------------------------------------------
3. WHAT ATTENTION SOLVES IN SEQUENCE-TO-SEQUENCE TASKS
--------------------------------------------------------------
Even with LSTMs, compressing an entire input sequence into a single fixed
vector (the final hidden state) loses information. Attention mechanism
solves this by allowing the model to "look back" at ALL input positions
when generating each output word.

Instead of relying only on the last hidden state, attention computes a
weighted sum of all hidden states, giving higher weights to the most
relevant parts of the input for the current prediction.

Example: In translation, when translating the word "cat" in the output,
attention helps the model focus on the corresponding word in the input
rather than treating all input words equally.

--------------------------------------------------------------
4. WHY TRANSFORMERS ARE IMPORTANT IN MODERN NLP AND GENERATIVE AI
--------------------------------------------------------------
Transformers (introduced in "Attention Is All You Need", 2017) replaced
RNNs entirely with a mechanism called Self-Attention. Key advantages:

a) Parallel Processing: Unlike RNNs that process words one-by-one,
   transformers process ALL words simultaneously, making training much faster.

b) Better Long-Range Dependencies: Self-attention directly connects every
   word to every other word, regardless of distance in the sentence.

c) Scalability: Transformers scale efficiently to massive datasets and
   model sizes, enabling models like GPT, BERT, and LLaMA.

d) Foundation for Generative AI: All modern large language models (ChatGPT,
   Claude, Gemini) are built on the transformer architecture. They can
   generate coherent text, translate languages, summarize documents, and
   answer questions because transformers excel at understanding context
   and relationships in text.

e) Transfer Learning: Pre-trained transformers (like BERT) can be fine-tuned
   on small datasets for specific tasks, democratizing NLP capabilities.

============================================================
GENERATING OUTPUT FILES
============================================================
Files generated:
  - model_evaluation.csv
  - sample_predictions.txt
Project complete!


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
