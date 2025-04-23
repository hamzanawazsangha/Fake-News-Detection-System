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
from huggingface_hub import hf_hub_download
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# Download NLTK punkt tokenizer (only once at startup)
nltk.download('punkt', quiet=True)

# Set page config
st.set_page_config(page_title="Fake News Detection", layout="wide")

@st.cache_resource
def load_resources():
    """Load all models and resources with caching"""
    resources = {
        'word2vec_model': None,
        'models': {},
        'accuracy_comparison': None,
        'classification_reports': {},
        'confusion_matrices': {},
        'roc_data': {},
        'le': LabelEncoder()
    }
    resources['le'].classes_ = np.array(['Fake', 'Real'])

    # Load Word2Vec model from Hugging Face
    try:
        model_path = hf_hub_download(
            repo_id="HamzaNawaz17/FakeNewsDetectionModel",
            filename="word2vec.model",
            cache_dir="models"
        )
        resources['word2vec_model'] = Word2Vec.load(model_path)
    except Exception as e:
        st.error(f"Failed to load Word2Vec model: {str(e)}")

    # Model loading function with error handling
    def load_model_file(filepath, model_name=""):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading {model_name}: {str(e)}")
            return None

    # Load machine learning models
    model_files = {
        "Logistic Regression": "Logistic Regression.pkl",
        "Naive Bayes": "Naive Bayes.pkl",
        "Random Forest": "Random Forest.pkl",
        "Support Vector Machine": "SVC.pkl",
        "XGBoost": "XGBOOST.pkl"
    }
    
    for name, file in model_files.items():
        resources['models'][name] = load_model_file(file, name)
    
    # Load Neural Network separately
    try:
        resources['models']["Neural Network"] = load_model("Fakenews.h5")
    except Exception as e:
        st.error(f"Error loading Neural Network: {str(e)}")

    # Load evaluation metrics
    metrics = {
        'accuracy_comparison': "accuracy_comparison_plot.pkl",
        'classification_reports': {
            "KNN": "classification_KNN.pkl",
            "Logistic Regression": "classification_LR.pkl",
            "Naive Bayes": "classification_NB.pkl",
            "Neural Network": "classification_Neural.pkl",
            "Random Forest": "classification_RF.pkl",
            "SVM": "classification_svc.pkl",
            "XGBoost": "classification_xgb.pkl"
        },
        'confusion_matrices': {
            "KNN": "confusion_KNN.pkl",
            "Logistic Regression": "confusion_LR.pkl",
            "Naive Bayes": "confusion_NB.pkl",
            "Neural Network": "confusion_Neural.pkl",
            "Random Forest": "confusion_RF.pkl",
            "SVM": "confusion_svc.pkl",
            "XGBoost": "confusion_xgb.pkl"
        },
        'roc_data': {
            "KNN": "roc_data_KNN.pkl",
            "Logistic Regression": "roc_data_LR.pkl",
            "Naive Bayes": "roc_data_NB.pkl",
            "Random Forest": "roc_data_RF.pkl",
            "Neural Network": "roc_data_neural.pkl",
            "SVM": "roc_data_svc.pkl",
            "XGBoost": "roc_data_xgb.pkl"
        }
    }

    for metric_type, files in metrics.items():
        if isinstance(files, dict):
            for name, file in files.items():
                loaded = load_model_file(file, f"{name} {metric_type}")
                if loaded is not None:
                    resources[metric_type][name] = loaded
        else:
            resources[metric_type] = load_model_file(files, metric_type)

    return (
        resources['word2vec_model'],
        resources['models'],
        resources['accuracy_comparison'],
        resources['classification_reports'],
        resources['confusion_matrices'],
        resources['roc_data'],
        resources['le']
    )

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load resources
word2vec_model, models, accuracy_comparison, classification_reports, confusion_matrices, roc_data, le = load_resources()

def preprocess_text(text):
    """Clean and tokenize text"""
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha()]

def get_sentence_embedding(sentence, model, vector_size=100):
    """Convert sentence to embedding vector"""
    if model is None:
        st.error("Word2Vec model not loaded properly")
        return np.zeros(vector_size)
    
    try:
        embeddings = [model.wv[word] for word in sentence if word in model.wv]
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(vector_size)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return np.zeros(vector_size)

def display_prediction_results(prediction, prediction_proba, model_name):
    """Display classification results in a structured way"""
    predicted_class = le.inverse_transform([prediction])[0]
    
    # Prediction header
    col1, col2 = st.columns(2)
    with col1:
        if predicted_class == "Fake":
            st.error(f"**Prediction:** {predicted_class} News")
        else:
            st.success(f"**Prediction:** {predicted_class} News")
    with col2:
        st.metric("Selected Model", model_name)
    
    # Confidence scores
    fake_prob = prediction_proba[0] * 100
    real_prob = prediction_proba[1] * 100
    
    st.write("### Confidence Scores")
    cols = st.columns(2)
    cols[0].metric("Fake Probability", f"{fake_prob:.2f}%")
    cols[1].metric("Real Probability", f"{real_prob:.2f}%")
    
    # Confidence visualization
    st.bar_chart(pd.DataFrame({
        'Class': ['Fake', 'Real'],
        'Probability': [fake_prob, real_prob]
    }).set_index('Class'))
    
    st.info("""
    **Note:** This is a machine learning model prediction and may not be 100% accurate. 
    Use critical thinking when evaluating news sources.
    """)

def main():
    st.title("Fake News Detection System")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Classifier", "Model Comparison", "About"])
    
    with tab1:
        st.header("News Article Classifier")
        st.write("Enter a news article text below to classify it as real or fake using different machine learning models.")
        
        user_input = st.text_area("News Article Text:", height=200)
        selected_model = st.selectbox("Select Model:", list(models.keys()))
        
        if st.button("Classify") and user_input:
            with st.spinner("Processing..."):
                try:
                    tokens = preprocess_text(user_input)
                    embedding = get_sentence_embedding(tokens, word2vec_model)
                    
                    if embedding is None:
                        st.error("Failed to generate embeddings")
                        return
                    
                    embedding = embedding.reshape(1, -1)
                    
                    if selected_model == "Neural Network":
                        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], 1)
                        prediction_proba = models[selected_model].predict(embedding)[0]
                        prediction = np.argmax(prediction_proba)
                        prediction_proba = [prediction_proba[0], prediction_proba[1]]
                    else:
                        model = models[selected_model]
                        prediction = model.predict(embedding)[0]
                        prediction_proba = model.predict_proba(embedding)[0]
                    
                    display_prediction_results(prediction, prediction_proba, selected_model)
                except Exception as e:
                    st.error(f"Classification error: {str(e)}")
    
    with tab2:
        st.header("Model Comparison")
        if accuracy_comparison:
            st.subheader("Model Accuracy Comparison")
            st.pyplot(accuracy_comparison)
            
            model_for_details = st.selectbox(
                "Select model to view detailed metrics:",
                list(classification_reports.keys()),
                index=1
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Classification Report")
                if model_for_details in classification_reports:
                    st.dataframe(pd.DataFrame(classification_reports[model_for_details]).transpose())
            
            with col2:
                st.subheader("Confusion Matrix")
                if model_for_details in confusion_matrices:
                    fig, ax = plt.subplots()
                    sns.heatmap(confusion_matrices[model_for_details], 
                               annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
            
            st.subheader("ROC Curve")
            if model_for_details in roc_data:
                st.pyplot(roc_data[model_for_details])
        else:
            st.warning("Evaluation metrics not loaded properly")
    
    with tab3:
        st.header("About This Project")
        st.markdown("""
        ### Fake News Detection System
        
        **Models Included:**
        - Logistic Regression
        - Naive Bayes
        - Random Forest
        - Support Vector Machine (SVM)
        - XGBoost
        - Neural Network
        
        **How it works:**
        1. Text preprocessing (tokenization, cleaning)
        2. Word2Vec embedding conversion
        3. Model prediction with confidence scores
        
        **Note:** The Word2Vec model is loaded from [Hugging Face Hub](https://huggingface.co/HamzaNawaz17/FakeNewsDetectionModel).
        """)

if __name__ == "__main__":
    main()
