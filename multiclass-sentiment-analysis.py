import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import re
import nltk
from nltk.tokenize import WhitespaceTokenizer, WordPunctTokenizer
from string import punctuation
import unidecode

# Download NLTK resources if needed
# nltk.download('stopwords')

class MulticlassSentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.classes = None
    
    def preprocess_text(self, texts):
        """
        Preprocess text through multiple steps:
        1. Remove stopwords
        2. Remove punctuation
        3. Remove accents
        4. Convert to lowercase
        5. Apply stemming
        """
        # Initialize tokenizers
        token_space = WhitespaceTokenizer()
        token_punct = WordPunctTokenizer()
        
        # Get stopwords
        try:
            stop_words = nltk.corpus.stopwords.words("portuguese")
        except:
            # Fallback if stopwords not available
            stop_words = []
        
        # Remove emoticons
        def remove_emoticons(text):
            emoticon_pattern = r'[:;=][\-\^]?[\)\(DPp@#$&|]'
            return re.sub(emoticon_pattern, '', text)
        
        # Step 1: Remove stopwords
        processed_texts = []
        for text in texts:
            new_text = []
            text_tokens = token_space.tokenize(remove_emoticons(text))
            for word in text_tokens:
                if word not in stop_words:
                    new_text.append(word)
            processed_texts.append(' '.join(new_text))
        
        # Step 2: Remove punctuation
        punct_list = list(punctuation)
        punct_stop_words = punct_list + stop_words
        
        processed_texts_2 = []
        for text in processed_texts:
            new_text = []
            text_tokens = token_punct.tokenize(text)
            for word in text_tokens:
                if word not in punct_stop_words:
                    new_text.append(word)
            processed_texts_2.append(' '.join(new_text))
        
        # Step 3: Remove accents
        processed_texts_3 = [unidecode.unidecode(text) for text in processed_texts_2]
        
        # Further process removing any remaining stopwords without accents
        stop_words_no_accent = [unidecode.unidecode(word) for word in punct_stop_words]
        
        processed_texts_4 = []
        for text in processed_texts_3:
            new_text = []
            # Step 4: Convert to lowercase
            text = text.lower()
            text_tokens = token_punct.tokenize(text)
            for word in text_tokens:
                if word not in stop_words_no_accent:
                    new_text.append(word)
            processed_texts_4.append(' '.join(new_text))
        
        # Step 5: Apply stemming
        try:
            stemmer = nltk.RSLPStemmer()
            processed_texts_5 = []
            for text in processed_texts_4:
                new_text = []
                text_tokens = token_punct.tokenize(text)
                for word in text_tokens:
                    new_text.append(stemmer.stem(word))
                processed_texts_5.append(' '.join(new_text))
            return processed_texts_5
        except:
            # Return non-stemmed version if stemmer not available
            return processed_texts_4
    
    def fit(self, df, text_column, label_column):
        """
        Train the sentiment analyzer model using the provided dataframe.
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing text and labels
        text_column (str): Name of the column containing the text
        label_column (str): Name of the column containing the labels
        """
        # Create a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Get unique classes and create a mapping
        self.classes = data[label_column].unique()
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        # Convert text data to numerical labels if needed
        if data[label_column].dtype == 'object':
            # Create a mapping from original labels to numeric values
            self.label_encoder = {label: i for i, label in enumerate(self.classes)}
            data['encoded_label'] = data[label_column].map(self.label_encoder)
        else:
            # If already numeric, just copy
            data['encoded_label'] = data[label_column]
            self.label_encoder = {i: label for i, label in enumerate(self.classes)}
        
        # Preprocess the text
        print("Preprocessing text data...")
        processed_texts = self.preprocess_text(data[text_column].values)
        data['processed_text'] = processed_texts
        
        # Split into train and test sets
        X = data['processed_text'].values
        y = data['encoded_label'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorize the text
        print("Vectorizing text data...")
        self.vectorizer = TfidfVectorizer()
        X_train_vect = self.vectorizer.fit_transform(X_train)
        X_test_vect = self.vectorizer.transform(X_test)
        
        # Train the model
        print("Training model...")
        self.model = MultinomialNB()
        self.model.fit(X_train_vect, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_vect)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Get detailed classification report
        print("\nClassification Report:")
        target_names = [f"{label} ({self.label_encoder[i]})" for i, label in enumerate(self.classes)]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return self
    
    def predict(self, texts):
        """
        Predict the sentiment class for the provided texts.
        
        Parameters:
        texts (list): List of text strings to classify
        
        Returns:
        list: Predicted class labels (original format)
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call fit() method first.")
        
        # Preprocess the text
        processed_texts = self.preprocess_text(texts)
        
        # Vectorize the text
        X_vect = self.vectorizer.transform(processed_texts)
        
        # Get numeric predictions
        numeric_predictions = self.model.predict(X_vect)
        
        # Convert back to original labels
        inverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
        original_predictions = [inverse_label_encoder[pred] for pred in numeric_predictions]
        
        return original_predictions
    
    def predict_proba(self, texts):
        """
        Predict class probabilities for the provided texts.
        
        Parameters:
        texts (list): List of text strings
        
        Returns:
        list of dict: Each dict maps class label to probability
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call fit() method first.")
        
        # Preprocess the text
        processed_texts = self.preprocess_text(texts)
        
        # Vectorize the text
        X_vect = self.vectorizer.transform(processed_texts)
        
        # Get probability predictions
        proba_predictions = self.model.predict_proba(X_vect)
        
        # Convert to dictionaries mapping class names to probabilities
        result = []
        inverse_label_encoder = {v: k for k, v in self.label_encoder.items()}
        
        for proba in proba_predictions:
            class_proba = {}
            for i, p in enumerate(proba):
                original_class = inverse_label_encoder[i]
                class_proba[original_class] = p
            result.append(class_proba)
        
        return result
    
    def save(self, model_path="sentiment_model", vectorizer_path="sentiment_vectorizer"):
        """Save the trained model and vectorizer to disk."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call fit() method first.")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'classes': self.classes
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"Model saved to {model_path} and {vectorizer_path}")
    
    def load(self, model_path="sentiment_model", vectorizer_path="sentiment_vectorizer"):
        """Load a trained model and vectorizer from disk."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']
                self.classes = model_data['classes']
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
                
            print(f"Model loaded successfully with {len(self.classes)} classes: {self.classes}")
            return self
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Example 1: Training with binary sentiment (like the original code)
    # Load dataset
    df = pd.read_csv("imdb-reviews-pt-br.csv")
    
    # Initialize and train the model
    analyzer = MulticlassSentimentAnalyzer()
    analyzer.fit(df, text_column='text_pt', label_column='sentiment')
    
    # Save the model
    analyzer.save("sentiment_model", "sentiment_vectorizer")
    
    # Make predictions
    texts = [
        "Este filme é maravilhoso, adorei cada minuto!",
        "Que decepção, perdi meu tempo assistindo isso."
    ]
    predictions = analyzer.predict(texts)
    probabilities = analyzer.predict_proba(texts)
    
    for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
        print(f"\nText {i+1}: {text}")
        print(f"Prediction: {pred}")
        print(f"Probabilities: {prob}")
    
    # Example 2: Training with multiple sentiment classes (if data were available)
    # This would work the same way but with more than 2 classes in the label_column
