from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
import base64
import io
import numpy as np

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except Exception as e:
            st.error(f"Error recognizing speech: {str(e)}")
            return None

def analyze_and_visualize(df):
    """Analyze and create visualizations for the dataset"""
    st.write("### Data Analysis and Visualizations")

    st.write("#### Basic Statistics")
    st.write(df.describe())
    
    tab1, tab2, tab3 = st.tabs(["Numerical Analysis", "Categorical Analysis", "Correlation Analysis"])
    
    with tab1:
        st.write("#### Numerical Columns Distribution")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            selected_num_col = st.selectbox("Select numerical column", numerical_cols, key="num_col")
            fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
            st.plotly_chart(fig, use_container_width=True)

            fig_box = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
            st.plotly_chart(fig_box, use_container_width=True)
    
    with tab2:
        st.write("#### Categorical Columns Analysis")
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            selected_cat_col = st.selectbox("Select categorical column", categorical_cols, key="cat_col")
            value_counts = df[selected_cat_col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f"Count of {selected_cat_col}",
                        labels={'x': selected_cat_col, 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)

            fig_pie = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Distribution of {selected_cat_col}")
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab3:
        st.write("#### Correlation Analysis")
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            fig = px.imshow(corr_matrix, 
                          title="Correlation Heatmap",
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
            
            if len(numerical_cols) > 1:
                st.write("#### Scatter Plot Matrix")
                top_corr_pairs = []
                for i in range(len(numerical_cols)):
                    for j in range(i+1, len(numerical_cols)):
                        corr = corr_matrix.iloc[i,j]
                        if abs(corr) > 0.5: 
                            top_corr_pairs.append((numerical_cols[i], numerical_cols[j], corr))
                
                if top_corr_pairs:
                    selected_pair = st.selectbox("Select feature pair", 
                                               [f"{pair[0]} vs {pair[1]} (corr: {pair[2]:.2f})" 
                                                for pair in top_corr_pairs])
                    selected_idx = [f"{pair[0]} vs {pair[1]} (corr: {pair[2]:.2f})" 
                                  for pair in top_corr_pairs].index(selected_pair)
                    x_col, y_col = top_corr_pairs[selected_idx][0], top_corr_pairs[selected_idx][1]
                    
                    fig_scatter = px.scatter(df, x=x_col, y=y_col, 
                                          title=f"Scatter Plot: {x_col} vs {y_col}")
                    st.plotly_chart(fig_scatter, use_container_width=True)

def csv_agent():
    df = None
    voice_enabled = st.sidebar.checkbox("Enable Voice", key="csv_voice")
    
    if 'csv_question' not in st.session_state:
        st.session_state.csv_question = "Generate me a report on this dataset"

    from langchain_experimental.agents.agent_toolkits import (
        create_pandas_dataframe_agent,
        create_csv_agent,
    )

    CSV_PROMPT_PREFIX = """
    First set the pandas display options to show all the columns,
    get the column names, then answer the question.
    """

    CSV_PROMPT_SUFFIX = """
    - **ALWAYS** before giving the Final Answer, try another method.
    Then reflect on the answers of the two methods you did and ask yourself
    if it answers correctly the original question.
    If you are not sure, try another method.
    FORMAT 4 FIGURES OR MORE WITH COMMAS.
    - If the methods tried do not give the same result,reflect and
    try again until you have two methods that have the same result.
    - If you still cannot arrive to a consistent result, say that
    you are not sure of the answer.
    - If you are sure of the correct answer, create a beautiful
    and thorough response using Markdown.
    - **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
    ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
    - **ALWAYS**, as part of your "Final Answer", explain how you got
    to the answer on a section that starts with: "\n\nExplanation:\n".
    In the explanation, mention the column names that you used to get
    to the final answer.
    """

    st.title("Database AI Agent with LangChain")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Upload your dataset in CSV format"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")

            agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                verbose=True,
                allow_dangerous_code=True,
                tools=[analyze_and_visualize]
            )
            
            st.success("File successfully uploaded and processed!")
            st.write("### Dataset Preview")
            st.write(df.head())
            
            st.write("### Ask a Question")
            
            if voice_enabled:
                if st.button("🎤 Record Question"):
                    voice_input = speech_to_text()
                    if voice_input:
                        st.session_state.csv_question = voice_input
                st.text_input("Your question:", value=st.session_state.csv_question, key="csv_question")
            else:
                st.text_input(
                    "Enter your question about the dataset:",
                    value=st.session_state.csv_question,
                    key="csv_question"
                )

            if st.button("Run Query"):
                question = st.session_state.csv_question
                
                if any(keyword in question.lower() for keyword in ['visualize', 'visualization', 'plot', 'graph', 'chart', 'display visual', 'show me the data']):
                    analyze_and_visualize(df)
                else:
                    QUERY = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
                    res = agent.invoke(QUERY)
                    st.write("### Final Answer")
                    st.markdown(res["output"])
                    
                    if voice_enabled:
                        audio_bytes = text_to_speech(res["output"])
                        autoplay_audio(audio_bytes)
                
                st.download_button("Download CSV Summary", df.describe().to_csv(), file_name="summary.csv")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV file to get started")

def pdf_agent():
    st.title("PDF AI Agent with LangChain")
    voice_enabled = st.sidebar.checkbox("Enable Voice", key="pdf_voice")
    
    if 'pdf_question' not in st.session_state:
        st.session_state.pdf_question = "What is the main content of the PDF?"

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload your PDF file"
    )

    if uploaded_file is not None:
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import FAISS
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain.chains import RetrievalQA

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )

            st.success("File successfully uploaded and processed!")
            st.write("### PDF Preview")
            st.write(f"Number of pages: {len(pages)}")
            st.write("First page preview:")
            st.write(pages[0].page_content[:500] + "...")

            st.write("### Ask a Question")  
            
            if voice_enabled:
                if st.button("🎤 Record Question"):
                    voice_input = speech_to_text()
                    if voice_input:
                        st.session_state.pdf_question = voice_input
                st.text_input("Your question:", value=st.session_state.pdf_question, key="pdf_question")
            else:
                st.text_input(
                    "Enter your question about the PDF:",
                    value=st.session_state.pdf_question,
                    key="pdf_question"
                )

            if st.button("Run Query"):
                question = st.session_state.pdf_question
                response = qa_chain.invoke({"query": question})
                st.write("### Final Answer")
                st.markdown(response["result"])
                
                if voice_enabled:
                    audio_bytes = text_to_speech(response["result"])
                    autoplay_audio(audio_bytes)
                
                st.download_button("Download PDF Summary", response["result"], file_name="summary.pdf")

            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")    
    else:
        st.info("Please upload a PDF file to get started")

def main():
    st.title("AI Agent with LangChain")     

    st.sidebar.title("Choose an Agent")
    page = st.sidebar.radio("Select a page", ["CSV Agent", "PDF Agent"])

    if page == "CSV Agent":
        csv_agent()
    elif page == "PDF Agent":
        pdf_agent()

if __name__ == "__main__":
    main()