from langchain.llms import GooglePalm
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()  # take environment variables from .env (especially openai api key)
api_key = os.environ["GOOGLE_API_KEY"]


# Create Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# # Initialize instructor embeddings using the Hugging Face model

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path = "faiss_index"
path = "codebasics_faqs.csv"



def create_vector_db():
    try:
        # Load data from FAQ sheet
        loader = CSVLoader(file_path=path, source_column="prompt")
        data = loader.load()

        if not data:
            print("No data loaded from CSV.")
            return

        # Create a FAISS instance for vector database from 'data'
        vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

        # Save vector database locally
        vectordb.save_local(vectordb_file_path)
        print(f"FAISS index created and saved successfully at {vectordb_file_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    
