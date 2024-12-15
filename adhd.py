import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

st.set_page_config(layout="wide")
@st.cache_data
def load_data():
    df = pd.read_csv("ADHD.csv")
    return df

@st.cache_data
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict.get(val, None)

def get_value(val, my_dict):
    return my_dict.get(val, None)
    


# Load your trained model (replace 'best_model.pkl' with your actual model file)
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Function to predict ADHD score
def predict_adhd_score(age, sex, previous_state, depression_total, alcohol_total,
                       aas1_total, university_performance, high_school_performance, anxiety_total,
                       aas_change):
    # Prepare input data
    input_data = np.array([[age, sex, previous_state, depression_total, alcohol_total,
                            aas1_total, university_performance, high_school_performance, anxiety_total,
                            aas_change]])
    # Predict ADHD score
    adhd_score = best_model.predict(input_data)
    return adhd_score[0]

# Sidebar setup

st.sidebar.image('logo.png', width=200) 
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction', 'Data Visualization', 'Model Information', 'Resources and Support', 'About'])  
# Add custom CSS for sidebar label
st.markdown(
    """
    <style>
    /* Style for the label text of the select box in the sidebar */
    [data-testid="stSidebar"] label {
        color: white;  /* Set the text color to white */
        
    </style>
    """,
    unsafe_allow_html=True
)


# Add custom CSS for sidebar and main content
st.markdown(
    """
    <style>
    /* Style for sidebar */
    [data-testid="stSidebar"] {
        background-color: #130C7C;  /* dark color */
        color: white;  /* Text color */
    }
    [data-testid="stSidebar"] .css-1pib6sb {  /* Style for sidebar headers */
        color: white;
    }
    [data-testid="stSidebar"] .css-h5rgaw {  /* Style for sidebar text */
        color: white;
    }

    /* Style for main content (right side) */
    [data-testid="stAppViewContainer"] {
        background-color: #E9E9E9;  /* Light gray color */
    }
    </style>
    """,
    unsafe_allow_html=True
)


if app_mode == 'Home':
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: darkblue;'>Welcome to the <span style='color: red;'>ADHD</span> Prediction Tool for University Students</h1>
    </div>
    """, unsafe_allow_html=True)
    
    
    
    st.markdown("""
    <div>
        <h3 style='color: Black;'>🎯 Purpose</h3>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
        This application aims to provide a convenient, data-driven way for university students to assess ADHD symptoms. By analyzing a set of behavioral, mental health, and academic factors, the app predicts an ADHD score that can help you better understand your mental well-being and its potential impact on your academic life.
       """)
    
    
    
    
    st.markdown("""
    <div>
        <h3 style='color: Black;'>🧠 What is <span style='color: red;'>ADHD</span>?</h3>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
        Attention Deficit Hyperactivity Disorder (ADHD) is a common neurodevelopmental disorder characterized by symptoms such as inattention, hyperactivity, and impulsivity. In a university setting, these symptoms may influence academic performance, affecting aspects like concentration, time management, and completion of tasks. Identifying ADHD early can help in managing these challenges and improving academic outcomes.

        ADHD can be frustrating for both teachers and students, but coming together to find strategies that work helps your student and your relationship blossom.
        """)



    st.subheader("Understanding and Supporting Students with ADHD")
    st.video("Understanding and Supporting Your Student With ADHD.mp4")
    st.markdown(
    """
    <style>
    video {
        height: 450px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    
    
    




elif app_mode == 'Prediction':
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: darkblue;'>💡 ADHD Score Predictor</h1>
    </div>
    """, unsafe_allow_html=True)
  
    st.markdown("""
    <div>
        <h4 style='color: Black;'>We need some information to predict the ADHD Score</h4>
    </div>
    """, unsafe_allow_html=True)


    # Create two columns
    col1, col2 = st.columns([1, 1])  # Adjust column width ratio if needed

    # Left column for input variables
    with col1:
        
        age = st.number_input("Age", min_value=0, max_value=100, step=1)
        sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        previous_state = st.radio("Previous State (0=No, 1=Yes)", options=[0, 1])
        depression_total = st.number_input("Depression Total", min_value=0)
        alcohol_total = st.number_input("Alcohol Total", min_value=0)
    with col2:

        aas1_total = st.number_input("AAS1 Total", min_value=0)
        university_performance = st.number_input("University Performance", min_value=0)
        
        # Adding space between the inputs
        st.markdown("<br>", unsafe_allow_html=True)

        high_school_performance = st.number_input("High School Performance", min_value=0)
        anxiety_total = st.number_input("Anxiety Total", min_value=0)
        aas_change = st.number_input("AAS Change", min_value=-100.0, value=0.00, format="%.4f")

    # Right column for output
    score = None
    
    if st.button("Predict ADHD Score"):
            score = predict_adhd_score(age, sex, previous_state, depression_total, alcohol_total,
                                       aas1_total, university_performance, high_school_performance,
                                       anxiety_total, aas_change)
    
    

    st.markdown("---")  # Horizontal line
    st.markdown("<h3 style='text-align: center; color: darkblue;'>📊 Prediction Results</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1]) 
    with col1:
        if score is not None:
            
            st.markdown(f"<h2 style='font-weight: bold; font-size: 24px;'>Predicted ADHD Score: {score:.2f}</h2>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if score > 24:
                st.error("Possible ADHD symptoms detected. Consult a professional.")
                st.image("adhd_positive.png", caption="Take Action for Better Mental Health", width=500)
            else:
                st.success("No significant ADHD symptoms detected. Stay mindful.")
                st.image("adhd_negative.png", caption="Keep Up the Good Work!", width=500)

    with col2:

        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        st.markdown("""
            
            - When the ADHD score is <span style="color:red; font-weight:bold;">greater than 24</span>, the student is considered <span style="color:red; font-weight:bold;">'At Risk'</span> for ADHD symptoms.
            - When the ADHD score is <span style="color:green; font-weight:bold;">below 24</span>, the student is considered <span style="color:green; font-weight:bold;">'Normal'</span> with no significant risk of ADHD.
       """, unsafe_allow_html=True)




elif app_mode == 'Data Visualization': 
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: darkblue;'>📈 Data Visualization</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    def load_data():
        df = pd.read_csv("ADHD.csv", encoding='latin1')  # Adjust encoding if needed
        return df

    # Call the load_data function
    df = load_data()

    # Rename columns
    df.rename(columns={
        'asrs1_total.y': 'adhd_total',
        'psy1004_grade': 'University_Performance',
        'bdi1_total': 'depression_Total',
        'matric_mark': 'HighSchool_performance',
        'bai1_total': 'Anexity_total',
        'audit1_total': 'Alcohol_total'
    }, inplace=True)

    # Create two columns for displaying graphs
    col1, col2 = st.columns(2)

    # Scatter Plot
    with col1:
        st.markdown("<h3 style='font-size: 16px;'>Scatter Plot: ADHD Symptoms vs University Performance</h3>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots()
        ax1.scatter(df['adhd_total'], df['University_Performance'], alpha=0.5)
        ax1.set_title('ADHD Symptoms vs University Performance')
        ax1.set_xlabel('ADHD Total')
        ax1.set_ylabel('University Performance')
        st.pyplot(fig1)

    # Violin Plot
    with col2:
        st.markdown("<h3 style='font-size: 16px;'>Violin Plot: ADHD Symptoms across Depression Levels</h3>", unsafe_allow_html=True)
       
        df['depression_category'] = pd.cut(
            df['depression_Total'], bins=[0, 10, 20, 30], labels=['Low', 'Medium', 'High']
        )
        fig2, ax2 = plt.subplots()
        sns.violinplot(data=df, x='depression_category', y='adhd_total', palette='muted', ax=ax2)
        ax2.set_title('ADHD Symptoms across Depression Levels')
        ax2.set_xlabel('Depression Level')
        ax2.set_ylabel('ADHD Total')
        st.pyplot(fig2)

    # Line Plot
    with col1:
        st.markdown("<h3 style='font-size: 16px;'>Line Plot: ADHD Symptoms vs University Performance by Age</h3>", unsafe_allow_html=True)
        df['age_group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50], labels=['<20', '20-30', '30-40', '40-50'])
        fig3, ax3 = plt.subplots()
        sns.lineplot(data=df, x='University_Performance', y='adhd_total', hue='age_group', markers=True, ax=ax3)
        ax3.set_title('ADHD Symptoms vs University Performance by Age')
        ax3.set_xlabel('University Performance')
        ax3.set_ylabel('ADHD Total Score')
        st.pyplot(fig3)

    # Box Plot
    with col2:
        st.markdown("<h3 style='font-size: 16px;'>Box Plot: Distribution of ADHD symptoms between different gender.</h3>", unsafe_allow_html=True)
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df, x='sex', y='adhd_total', ax=ax4)
        ax4.set_title('ADHD Symptoms by Gender')
        ax4.set_xlabel('Gender')
        ax4.set_ylabel('ADHD Total')
        st.pyplot(fig4)

    # Distribution Plot for aas_change 
    with col1:
        st.markdown("<h3 style='font-size: 16px;'>aas_change shows the frequency and spread of changes in academic adjustment</h3>", unsafe_allow_html=True)
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        sns.histplot(df['aas_change'].dropna(), kde=True, bins=30, ax=ax6)
        ax6.set_title('Distribution of aas_change Feature')
        ax6.set_xlabel('aas_change')
        ax6.set_ylabel('Frequency')
        st.pyplot(fig6)

    # Correlation Heatmap
    with col2:
        st.markdown("<h3 style='font-size: 16px;'>Correlation Heatmap: ADHD Symptoms and Other Variables</h3>", unsafe_allow_html=True)
        correlation_matrix = df[['adhd_total', 'University_Performance', 'HighSchool_performance', 
                                 'depression_Total', 'Anexity_total', 'Alcohol_total']].corr()
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax7)
        ax7.set_title('Correlation Heatmap of ADHD Symptoms and Other Variables')
        st.pyplot(fig7)






elif app_mode == 'Model Information':
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: darkblue;'>⚙️ Model Information</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
        The ADHD prediction model behind this app uses machine learning to analyze various factors that may indicate ADHD symptoms.
        
        """)

    st.write("""
        By entering details about demographics, mental health, and academic performance, the app generates an ADHD score. This score helps students understand potential ADHD symptoms and identify areas where support may be needed.
        
        """)
    
    st.markdown("""
    <div>
        <h3 style='color: Black;'>🔬 Model Training Details</h3>
    </div>
    """, unsafe_allow_html=True)

    st.subheader('Dataset')
    st.write(
    """
    The model was trained using the dataset from Kaggle's ADHD and Mental Health dataset. 
    The dataset includes information on academic performance, behavioral data, and mental health factors.
    """
)

    # Adding a link to the dataset
    st.markdown(
        "[Kaggle's ADHD and Mental Health Dataset](https://www.kaggle.com/datasets/xyz/adhd-mental-health)"
    )
    
    st.subheader('Algorithms Used')
    st.write(
        """
        The model uses a combination of the following supervised learning algorithms to make predictions based on the input data:
        - Linear Regression
        - Decision Tree
        - Random Forest
        - Support Vector Regressor
        - Gradient Boosting
        """
    )
    
    st.subheader('Training Process')
    st.write(
        """
        The data was split into training and testing sets to evaluate model performance. 
        10-fold cross-validation was applied to ensure that the model generalizes well to new, unseen data.
        """
    )
    
    st.subheader('Performance Evaluation')
    st.write(
        """
        As this is a regression model, the following evaluation metrics were used to assess model performance:
        - Mean Squared Error (MSE)
        - Mean Absolute Error (MAE)
        - R-squared (R²) Score
        """
    )
    
    st.image("evaluation.png", caption="Evaluation Metrics", width=500)
    st.write(
        """
        In evaluation, the best model is selected based on the combination of these metrics. For instance, if Linear Regression has the lowest MSE, MAE and the highest R2 score among all models, it would be considered the best-performing model.

        """
    )


elif app_mode == 'Resources and Support':
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: darkblue;'>📚 Resources and Support</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div>
        <h3 style='color: Black;'> 🎓 Managing ADHD in University Life</h3>
    </div>
    """, unsafe_allow_html=True)

    
    st.write("""If you suspect that ADHD symptoms might be affecting your academic performance, you’re not alone. Here are some strategies and resources to help you thrive:
        """)

    st.write("""
        - Time Management Tools: Use apps like Trello, Notion, or Google Calendar to organize tasks and set reminders.
        - Study Techniques: Break tasks into smaller, manageable parts and use the Pomodoro technique to stay focused.
        - Support Groups: Join university or online support groups for students with ADHD to share experiences and tips.
        - Mental Health Services: Reach out to your university’s counseling center for professional advice and support.
    """)

    st.markdown("""
    <div>
        <h3 style='color: Black;'>🌐 Helpful Online Resources</h3>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    - <a href="https://chadd.org" target="_blank"><b>CHADD (Children and Adults with ADHD):</b></a> Provides educational resources, webinars, and support for individuals with ADHD.
    - <a href="https://www.additudemag.com/" target="_blank"><b>ADDitude Magazine:</b></a> Offers expert advice on managing ADHD at school and beyond.
    - <a href="https://www.adhdfoundation.org.uk/" target="_blank"><b>ADHD Foundation:</b></a> A charity providing resources and support for individuals with ADHD.
    """, unsafe_allow_html=True)

    st.markdown("""
    <div>
        <h3 style='color: Black;'>👨‍👩‍👧‍👦 Tips for Parents and Educators</h3>
    </div>
    """, unsafe_allow_html=True)
           
                
    st.write("""
        - Foster a positive, encouraging environment.
        - Focus on strengths rather than just challenges.
        - Use visual aids and structured routines to support learning.
 
    """)
             





elif app_mode == 'About':
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: darkblue;'>About</h1>
    </div>
    """, unsafe_allow_html=True)
    

    st.markdown("""
    <div>
        <h3 style='color: Black;'>💡 The Motivation Behind This Application</h3>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
        This project stems from my individual research initiative aimed at understanding the impact of ADHD symptoms on university students’ academic performance. By combining insights from behavioral science and machine learning, I aim to provide a tool that helps students better understand and manage their mental well-being.
         """)
    


    st.markdown("""
    <div>
        <h3 style='color: Black;'>🎯 My Mission</h3>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
        My goal is to bridge the gap between awareness and action by making ADHD screening accessible, data-driven, and actionable. Ultimately, I hope to promote early identification and provide students with the resources needed to excel academically and personally.
              """)
    

    st.markdown("""
    <div>
        <h3 style='color: Black;'>🛠️ Development</h3>
    </div>
                
    """, unsafe_allow_html=True)

    st.write("""
       This app was developed as part of a my research project, leveraging publicly available data and machine learning techniques to create a reliable predictive model.
             """)
    

    st.markdown("""
    <div>
        <h3 style='color: Black;'>🌟 Acknowledgments </h3>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
        We would like to thank the educators, mental health professionals, and university students who contributed to this project, as well as the open data community for making this research possible. 
             """)
    
    st.write("""
        - For more details about this project, feel free to contact me at isharaupendrika22@gmail.com.
    """)


      # Your photo and introduction
    col1, col2, col3 = st.columns([1, 1, 1])  # Center-align photo and details
    with col3:
        st.image("ishara.jpeg", caption="Ishara Upendrika - Undergraduate", width=150)


            
