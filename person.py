import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import textract
import docx
import io

class TrainModel:
    def __init__(self):
        self.mul_lr = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.training()

    def training(self):
        df = pd.read_csv("C:\\Users\\anuha\\Downloads\\archive (12)\\train.csv")
        arr = df.values

        for i in range(len(arr)):
            arr[i][0] = 1 if arr[i][0] == "Male" else 0

        data = pd.DataFrame(arr)

        maindata = data.drop(7,axis=1)
        mainarray = maindata.values

        personality = data[7]
        train_p = personality.values

        self.mul_lr = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
        self.mul_lr.fit(mainarray, train_p)

    def testing(self, test_data):
        try:
            #predict_test = list(map(int, test_data))
            p_pred = self.mul_lr.predict([test_data])
            return p_pred[0]
        except Exception as e:
            print(f"Error during testing: {e}")
        return None

def predict_person():
    st.sidebar.title("Personality Prediction using CV/Resume Analysis")
    sName = st.sidebar.text_input("Your Name")
    age = st.sidebar.text_input("Your Age")
    gender = st.sidebar.radio("You have been identified as", ["Male", "Female"])
    file_path = st.sidebar.file_uploader("Attach your CV (PDF/DOCX)", type=["pdf", "docx"])

     # Define file_contents outside of the block
    file_contents = None

    if file_path:
        file_contents = file_path.read()

    if st.sidebar.button("Analyze and Predict"):
        if not file_path:
            st.error("Please select a CV file")
        else:
            try:
                # Preprocess the test data
                df_test = pd.read_csv("C:\\Users\\anuha\\Downloads\\archive (12)\\test.csv")
                test_data = df_test[['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']]
                test_data = test_data.values  # Convert to numpy array
                #file_contents = file_path.read()
                personality_values = analyze_cv_and_predict(gender, age, file_path.name, file_contents)
                if personality_values is not None:
                    prediction_result(sName, age, personality_values)
            except Exception as e:
                st.error(f"Error: {e}")

def analyze_cv_and_predict(gender, age, file_name, file_contents):
    try:
        # Check the file type and extract text accordingly
        if file_name.endswith('.pdf'):
            cv_text = textract.process(file_contents, method='pdftotext')
            cv_text = cv_text.decode("utf-8", errors="replace")
        elif file_name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_contents))
            cv_text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            st.error("Unsupported file format. Please upload a PDF or DOCX file.")
            return None

        scores = {
            'openness': 4,
            'neuroticism': 4,
            'conscientiousness': 4,
            'agreeableness': 4,
            'extraversion': 4,
            'Gender': 4,
            'Age' : 4
        }
        keywords = {
            'openness': ['openness', 'inventive', 'curious', 'variety', 'imaginative', 'creative', 'adventurous',
                         'innovative', 'unconventional'],
            'neuroticism': ['neuroticism', 'sensitive', 'nervous', 'emotional', 'anxious', 'moody', 'temperamental',
                            'fragile', 'vulnerable'],
            'conscientiousness': ['conscientiousness', 'organized', 'disciplined', 'diligent', 'dependable', 'methodical',
                                  'efficient', 'precise'],
            'agreeableness': ['agreeableness', 'friendly', 'compassionate', 'kind', 'helpful', 'sympathetic', 'tolerant',
                              'courteous', 'empathetic'],
            'extraversion': ['extraversion', 'outgoing', 'energetic', 'sociable', 'enthusiastic', 'assertive', 'vibrant',
                             'gregarious', 'lively'],
        }


        for trait, trait_keywords in keywords.items():
            for keyword in trait_keywords:
                if keyword in cv_text.lower():
                    scores[trait] = min(scores[trait] + 1, 7)

        personality_values = [scores[trait] for trait in
                              ['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']]

        st.success("CV analysis completed successfully.")
        return personality_values

    except Exception as e:
        st.error(f"Error: {e}")
        return None

def prediction_result(name, age, personality_values):
    st.title("Predicted Personality")
    applicant_data = {"Name of User": name, "Age": age}

    st.write("\n**ENTERED DATA**\n")
    st.write(applicant_data, personality_values)

    model = TrainModel()
    personality = model.testing(personality_values)

    st.write("\n**PERSONALITY PREDICTED**\n")
    st.write(personality)

    st.write("\n**TERMS MEAN**\n")
    st.write("""
    OPENNESS TO EXPERIENCE  - (inventive/curious vs. consistent/cautious).
    Appreciation for art, emotion, adventure, unusual ideas, curiosity, and variety of experience.

    CONSCIENTIOUSNESS - (efficient/organized vs. easy-going/careless).
    A tendency to show self-discipline, act dutifully, and aim for achievement;
    planned rather than spontaneous behavior.

    EXTRAVERSION - (outgoing/energetic vs. solitary/reserved).
    Energy, positive emotions, urgency, and the tendency to seek stimulation
    in the company of others.

    AGREEABLENESS - (friendly/compassionate vs. cold/unkind).
    A tendency to be compassionate and cooperative rather than suspicious
    and antagonistic towards others.

    NEUROTICISM - (sensitive/nervous vs. secure/confident).
    A tendency to experience unpleasant emotions easily, such as anger,
    anxiety, depression, or vulnerability.
    """)

if __name__ == "__main__":
    model = TrainModel()
    model.training()
    predict_person()
#streamlit run person.py