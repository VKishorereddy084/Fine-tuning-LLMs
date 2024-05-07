import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import os
import pandas as pd

# Define prompt templates
prompt_template_questions = """
You are an expert in creating practise questions based on study material.
Your goal is to prepare a set of questions. You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the end user. Make sure not to lose any important information.

QUESTIONS:
"""

PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])

refine_template_questions = """
You are an expert in creating questions based on study material.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.

QUESTIONS:
"""

REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)

# Request file path input from the user
file_path = input("Please enter the path to the PDF file: ")

# Check if file_path is valid
if file_path and os.path.isfile(file_path):
    # Load data from the uploaded PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Combine text from Document into one string for question generation
    text_question_gen = ''.join(page.page_content for page in data)
    
    # Initialize Text Splitter for question generation
    text_splitter_question_gen = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)
    text_chunks_question_gen = text_splitter_question_gen.split_text(text_question_gen)
    docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]
    
    # Initialize Large Language Model for question generation
    llm_question_gen = LlamaCpp(
        streaming=True,
        model_path="zephyr-7b-alpha.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1, 
        verbose=True,
        n_ctx=4096
    )

    # Initialize question generation chain
    question_gen_chain = load_summarize_chain(llm=llm_question_gen, chain_type="refine", verbose=True,
                                              question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)
    questions = question_gen_chain.run(docs_question_gen)

    # Initialize Large Language Model for answer generation
    llm_answer_gen = LlamaCpp(
        streaming=True,
        model_path="zephyr-7b-alpha.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1, 
        verbose=True,
        n_ctx=4096
    )

    # Create vector database for answer generation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vector_store = Chroma.from_documents(docs_question_gen, embeddings)
    answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever(k=2))
    
    # Answer questions and compile results
    question_answer_pairs = [(question, answer_gen_chain.run(question)) for question in questions.split("\n")]

    # Save results to a CSV file
    answers_dir = os.path.join(tempfile.gettempdir(), "answers")
    os.makedirs(answers_dir, exist_ok=True)
    qa_df = pd.DataFrame(question_answer_pairs, columns=["Question", "Answer"])
    csv_file_path = os.path.join(answers_dir, "questions_and_answers.csv")
    qa_df.to_csv(csv_file_path, index=False)

    print(f"Questions and answers saved to {csv_file_path}")

else:
    print("Invalid file path. Please make sure the file exists.")
