import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import SemanticSimilarityExampleSelector, FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType
import speech_recognition as sr
from io import StringIO

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="Ask MySQL", layout="wide")
st.title("Ask MySQL")

# Database configuration
DB_CONFIG = {
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "4082"),
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "hr_management"),
    "port": int(os.getenv("DB_PORT", 3306))
}

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Validate configuration
if not all(DB_CONFIG.values()) or not GROQ_API_KEY:
    st.error("Please provide all required configuration in the .env file.")
    st.stop()

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It", streaming=True)

llm = get_llm()

# Database configuration
@st.cache_resource(ttl=7200)
def configure_db():
    # Use SQLite for in-memory database
    connection_string = f"sqlite:///temp_db.sqlite"
    engine = create_engine(connection_string)
    return engine

db_engine = configure_db()

# SQL Agent configuration
@st.cache_resource
def get_sql_agent():
    toolkit = SQLDatabaseToolkit(db=SQLDatabase(db_engine), llm=llm)
    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

agent = get_sql_agent()

# Example selector and prompt configuration
@st.cache_resource
def configure_example_selector():
    from few_shots import few_shots
    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(texts=to_vectorize, embedding=embeddings)
    return SemanticSimilarityExampleSelector(vectorstore=vectorstore, k=2)

example_selector = configure_example_selector()

mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURDATE() function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: Query to run with no pre-amble
SQLResult: Result of the SQLQuery
Answer: Final answer here

No pre-amble.
"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    ),
    prefix=mysql_prompt,
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"],
)

chain = SQLDatabaseChain(llm=llm, database=SQLDatabase(db_engine), prompt=few_shot_prompt, verbose=False)

# CSV Upload
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        # Save DataFrame to SQLite database
        df.to_sql(name=os.path.splitext(uploaded_file.name)[0], con=db_engine, if_exists='replace', index=False)
        st.success(f"{uploaded_file.name} has been uploaded and processed.")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey! How can I help you?"}]

def display_chat():
    for msg in st.session_state.messages:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            st.markdown(f"**You:** {content}")
        elif role == 'assistant':
            st.markdown(f"**Bot:** {content}")
        else:
            st.markdown(f"**{role.capitalize()}:** {content}")

display_chat()

# Voice recognition function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Please speak now...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            return query
        except sr.UnknownValueError:
            st.error("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return ""

# User interaction
query = st.text_input('Ask me a SQL question...', '')

if st.button('Speak your query'):
    query = recognize_speech()
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.markdown(f"**You:** {query}")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f"**You:** {query}")

    with st.spinner("Generating response..."):
        try:
            # Pass the query as a keyword argument
            res = agent.run(input=query)
            st.session_state.messages.append({'role': 'assistant', 'content': res})
            st.markdown(f"**Bot:** {res}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
