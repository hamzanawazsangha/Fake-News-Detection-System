import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Download NLTK punkt tokenizer
nltk.download('punkt')

# Set page config
st.set_page_config(page_title="Fake News Detection", layout="wide")

# Load all models and evaluation metrics
@st.cache_resource
def load_resources():
    # Load Word2Vec model (you'll need to save this separately)
    word2vec_model = Word2Vec.load("word2vec.model") if hasattr(st, 'word2vec_model') else None
    
    # Load machine learning models
    models = {
        "Logistic Regression": pickle.load(open("Logistic Regression.pkl", "rb")),
        "Naive Bayes": pickle.load(open("Naive Bayes.pkl", "rb")),
        "Random Forest": pickle.load(open("Random Forest.pkl", "rb")),
        "Support Vector Machine": pickle.load(open("SVC.pkl", "rb")),
        "XGBoost": pickle.load(open("XGBOOST.pkl", "rb")),
        "Neural Network": load_model("Fakenews.h5")
    }
    
    # Load accuracy comparison
    accuracy_comparison = pickle.load(open("accuracy_comparison_plot.pkl", "rb"))
    
    # Load classification reports
    classification_reports = {
        "KNN": pickle.load(open("classification_KNN.pkl", "rb")),
        "Logistic Regression": pickle.load(open("classification_LR.pkl", "rb")),
        "Naive Bayes": pickle.load(open("classification_NB.pkl", "rb")),
        "Neural Network": pickle.load(open("classification_Neural.pkl", "rb")),
        "Random Forest": pickle.load(open("classification_RF.pkl", "rb")),
        "SVM": pickle.load(open("classification_svc.pkl", "rb")),
        "XGBoost": pickle.load(open("classification_xgb.pkl", "rb"))
    }
    
    # Load confusion matrices
    confusion_matrices = {
        "KNN": pickle.load(open("confusion_KNN.pkl", "rb")),
        "Logistic Regression": pickle.load(open("confusion_LR.pkl", "rb")),
        "Naive Bayes": pickle.load(open("confusion_NB.pkl", "rb")),
        "Neural Network": pickle.load(open("confusion_Neural.pkl", "rb")),
        "Random Forest": pickle.load(open("confusion_RF.pkl", "rb")),
        "SVM": pickle.load(open("confusion_svc.pkl", "rb")),
        "XGBoost": pickle.load(open("confusion_xgb.pkl", "rb"))
    }
    
    # Load ROC data
    roc_data = {
        "KNN": pickle.load(open("roc_data_KNN.pkl", "rb")),
        "Logistic Regression": pickle.load(open("roc_data_LR.pkl", "rb")),
        "Naive Bayes": pickle.load(open("roc_data_NB.pkl", "rb")),
        "Random Forest": pickle.load(open("roc_data_RF.pkl", "rb")),
        "Neural Network": pickle.load(open("roc_data_neural.pkl", "rb")),
        "SVM": pickle.load(open("roc_data_svc.pkl", "rb")),
        "XGBoost": pickle.load(open("roc_data_xgb.pkl", "rb"))
    }
    
    le = LabelEncoder()
    le.classes_ = np.array(['Fake', 'Real'])  # Assuming these are the classes
    
    return word2vec_model, models, accuracy_comparison, classification_reports, confusion_matrices, roc_data, le

word2vec_model, models, accuracy_comparison, classification_reports, confusion_matrices, roc_data, le = load_resources()

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    return tokens

# Function to get sentence embeddings
def get_sentence_embedding(sentence, model, vector_size=100):
    embeddings = [model.wv[word] for word in sentence if word in model.wv]
    if len(embeddings) == 0:
        return np.zeros(vector_size)
    return np.mean(embeddings, axis=0)

# Streamlit app
def main():
    st.title("Fake News Detection System")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Classifier", "Model Comparison", "About"])
    
    with tab1:
        st.header("News Article Classifier")
        st.write("""
        Enter a news article text below to classify it as real or fake using different machine learning models.
        """)
        
        # Input text
        user_input = st.text_area("News Article Text:", height=200, key="input_text")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model:",
            list(models.keys()),
            index=0
        )
        
        if st.button("Classify"):
            if user_input:
                # Preprocess and vectorize the input
                tokens = preprocess_text(user_input)
                embedding = get_sentence_embedding(tokens, word2vec_model)
                embedding = embedding.reshape(1, -1)
                
                # Make prediction
                if selected_model == "Neural Network":
                    # Reshape for neural network
                    embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], 1)
                    prediction_proba = models[selected_model].predict(embedding)[0]
                    prediction = np.argmax(prediction_proba)
                    prediction_proba = [prediction_proba[0], prediction_proba[1]]  # Convert to 2D array
                else:
                    prediction = models[selected_model].predict(embedding)
                    prediction_proba = models[selected_model].predict_proba(embedding)[0]
                
                # Display results
                st.subheader("Prediction Results")
                predicted_class = le.inverse_transform([prediction])[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    if predicted_class == "Fake":
                        st.error(f"**Prediction:** {predicted_class} News")
                    else:
                        st.success(f"**Prediction:** {predicted_class} News")
                
                with col2:
                    st.metric("Selected Model", selected_model)
                
                st.write("### Confidence Scores")
                fake_prob = prediction_proba[0] * 100
                real_prob = prediction_proba[1] * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fake Probability", f"{fake_prob:.2f}%")
                with col2:
                    st.metric("Real Probability", f"{real_prob:.2f}%")
                
                # Show confidence bar
                confidence_data = pd.DataFrame({
                    'Class': ['Fake', 'Real'],
                    'Probability': [fake_prob, real_prob]
                })
                
                st.bar_chart(confidence_data.set_index('Class'))
                
                st.write("""
                **Note:** This is a machine learning model prediction and may not be 100% accurate. 
                Use critical thinking when evaluating news sources.
                """)
            else:
                st.warning("Please enter some text to classify.")
    
    with tab2:
        st.header("Model Comparison")
        st.write("Compare the performance of different models used in this system.")
        
        # Display accuracy comparison
        st.subheader("Model Accuracy Comparison")
        st.pyplot(accuracy_comparison)
        
        # Select model for detailed metrics
        model_for_details = st.selectbox(
            "Select model to view detailed metrics:",
            list(classification_reports.keys()),
            index=1
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Report")
            st.table(pd.DataFrame(classification_reports[model_for_details]).transpose())
        
        with col2:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrices[model_for_details], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        # ROC Curve
        st.subheader("ROC Curve")
        roc_fig = roc_data[model_for_details]
        st.pyplot(roc_fig)
    
    with tab3:
        st.header("About This Project")
        st.write("""
        ### Fake News Detection System
        
        This application uses machine learning and deep learning models to classify news articles as real or fake.
        
        **Models Included:**
        - Logistic Regression
        - Naive Bayes
        - Random Forest
        - Support Vector Machine (SVM)
        - XGBoost
        - Neural Network (Deep Learning)
        
        **Features:**
        - Text classification using state-of-the-art models
        - Model performance comparison
        - Detailed evaluation metrics for each model
        - Confidence scores for predictions
        
        **How it works:**
        1. The system preprocesses the input text (tokenization, lowercasing, etc.)
        2. Converts the text to numerical features using Word2Vec embeddings
        3. Uses the selected model to make a prediction
        4. Returns the classification with confidence scores
        
        The models were trained on a dataset of thousands of labeled real and fake news articles.
        """)

if __name__ == "__main__":
    main()
