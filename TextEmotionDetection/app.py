'''import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model
pipe_lr = joblib.load(open("C:\\Users\\gayatri\\Desktop\\Text-Emotion-Detection-project\\Text Emotion Detection\\model\\text_emotion.pkl", "rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results[0]  # Selecting the first class probability for simplicity

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    # Allow the user to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.write("Uploaded Dataset:")
        st.write(df)
        
        # Check if 'Clean_Text' column exists
        if 'Clean_Text' not in df.columns:
            st.error("The 'Clean_Text' column is missing in the dataset.")
            return

        # Make predictions for each text in the dataset
        df['predictions'] = df['Clean_Text'].apply(predict_emotions)
        df['probabilities'] = df['Clean_Text'].apply(get_prediction_proba)

        # Display predictions and probabilities
        st.write("Predictions:")
        st.write(df['predictions'])

        st.write("Prediction Probabilities:")
        st.write(df['probabilities'])

        # Display visuals (you can customize this part based on your needs)
        st.write("Visualization:")
        proba_df = pd.DataFrame(df['probabilities'].tolist(), columns=pipe_lr.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["emotions", "probability"]
        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
        st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
'''
import streamlit as st
import pandas as pd
import altair as alt

import joblib
import os

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model using a relative path
model_path = os.path.join(current_dir, "model", "text_emotion.pkl")
pipe_lr = joblib.load(open(model_path, "rb"))
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def display_emotion_rows(df, selected_emotion):
    # Display rows of text for the selected emotion
    st.write(f"Rows of Text for {selected_emotion.capitalize()}:")
    selected_rows = df[df['predictions'] == selected_emotion]['Text']
    st.write(selected_rows)

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    # Allow the user to upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.write("Uploaded Dataset:")
        st.write(df)

        # Check if 'Text' column exists
        if 'Text' not in df.columns:
            st.error("The 'Text' column is missing in the dataset.")
            return

        # Fill missing values in the 'Text' column with an empty string
        df['Text'] = df['Text'].fillna('')

        # Make predictions for each text in the dataset
        df['predictions'] = df['Text'].apply(predict_emotions)

        # Display total number of reviews
        st.write(f"Total Number of Rows: {len(df)}")

        # Display count of all emotions
        st.write("Count of Emotions:")
        emotion_counts = df['predictions'].value_counts()
        st.write(emotion_counts)

        # Representation plot of emotions and their counts
        st.write("Representation Plot of Emotions:")
        chart = alt.Chart(df).mark_bar().encode(
            x='predictions',
            y='count()',
            color='predictions'
        ).properties(width=600, height=400)
        st.altair_chart(chart)

        # Display buttons for each emotion in the sidebar
        selected_emotion = st.sidebar.selectbox("Select Emotion", list(emotions_emoji_dict.keys()))

        if st.sidebar.button("Show Emotion Text"):
            # Show emotion text on another page
            st.write(f"Emotion Text for {selected_emotion.capitalize()}:")
            display_emotion_rows(df, selected_emotion)

if __name__ == '__main__':
    main()
