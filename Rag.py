import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from wordllama import WordLlama
import json
import tiktoken
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # type: ignore
import logging

logging.getLogger("faiss").setLevel(logging.ERROR)

############## RAG CODE ###################
def store_vectors(merged_output_file_path, file_name):

  #used to uplaod the pdf
  loader_py = PyPDFLoader(merged_output_file_path)
  pages_py = loader_py.load()

  print(pages_py[0].page_content)

  #to spefcify how we want to split our file
  text_splitter = RecursiveCharacterTextSplitter(
      #this is the size of each split
      chunk_size=1000,
      #this is how much splits overlap which each other
      chunk_overlap=150,
      separators=["."],
      length_function=len
  )

  docs = text_splitter.split_documents(pages_py)
  
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

  #to transfrom our chunks to embeddings
  vectorstore = FAISS.from_documents(docs, embedding=embeddings)

  vectorstore.save_local(file_name + "_index")

def rag(question, questions_file_path, file_name):
  #this function show us how we want to retrive our chunks
  #k = 3 means we retrive the three top chunks
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

  vectorstore = FAISS.load_local(
    f"{file_name}_index", embeddings, allow_dangerous_deserialization=True
  )

  retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

  with open("output.txt", "w", encoding="utf-8") as f:
    for i in range(len(retriever.invoke(question))):
      f.write(retriever.invoke(question)[i].page_content)

  #we use gpt-4o here
  llm = ChatGoogleGenerativeAI(temperature=0.7, model="models/gemma-3n-e4b-it")

  #The standard template fed to the rag
  template = """
            You are an intelligent information retrieval assistant. Your job is to answer the user's query using the information provided in the context below.

            **Context Source**: This information comes from a school book.

            **Important Instructions**:
            - Only use the provided context to answer the query.
            - Quote directly from the context if helpful.
            - Keep your answer concise, factual, and directly related to the query.
            - Only answer in the arabic language

            **User Query**:
            {query}
  
            **Context**:
            {context}

            **Your Answer**:
            """

  prompt = ChatPromptTemplate.from_template(template)

#   #this is the chain, first the input is modified by the template and after that it's passed to the LLM to get the output
  chain = (
      {"context":retriever,
      "query":RunnablePassthrough()}
      |  prompt  | llm | StrOutputParser()
  )
  AI_output = chain.invoke(question)

  # this is the structure of the output data
  new_data = {
      'Student Question': [question],
      'AI Answer': [AI_output],
  }

  #Each output is saved in the a csv file in case the teacher want's to modify some answers
  new_df = pd.DataFrame(new_data)
  new_df.to_csv(questions_file_path, mode='a', index=False, header=False)
  
  return(AI_output)

# ################## CHECK TEACHER'S ANSWERS #################
def check_teacher(question, questions_file_path):
  similarity_degree = 0.8

  wl = WordLlama.load()

  questions_data_frame = pd.read_csv(questions_file_path)
  a, b = questions_data_frame.shape
  similarities = []

  for i in range(a):
    if(str(questions_data_frame.iloc[i,2]) != "nan"):
      similarities.append(wl.similarity(question, questions_data_frame.iloc[i,0]))
    else:
      similarities.append(0)

  if(len(similarities) > 0):
    best = ""
    for similarity in similarities:
      index = similarities.index(similarity)
      if(similarity > similarity_degree and str(questions_data_frame.iloc[index,2]) != "nan"):
        best = questions_data_frame.iloc[index, 2]
    
    if(best != ""):
      return(best)

  return(False)

################### TO CHECK REMAINING TOKENS #################
def check_tokens(business_remaining_number_of_tokens, end_user_remaining_tokens, question):
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
  length = len(encoding.encode(question))
  if(business_remaining_number_of_tokens > length and end_user_remaining_tokens > length):
    return length
  else:
    return False


############### MAIN CODE ##########################

if __name__ == "__main__":

    #@@@@@@@@@@@@@@@@@@@@@@
    #PATH For JSON VARRIBLE CHANGE THEM ACCORDING TO YOUR SYSTEM
    json_input_path = "C:/Users/algez/Downloads/Educational_rag/json_input.json"
    json_output_path = "C:/Users/algez/Downloads/Educational_rag/json_output.json"

    with open(json_input_path, "r", encoding="utf-8") as file:
        json_input = json.load(file)

    question = json_input["query"]
    query_type = "text"
    business_id = "teaching_platform_123"
    grade = json_input['context']['education_level']['grade']
    semester = json_input['context']['education_level']['semester']
    subject = json_input['context']['subject']
    business_remaining_number_of_tokens = json_input['business_remaining_number_of_tokens']
    end_user_remaining_tokens= json_input['end_user_remaining_tokens']
    file_name = json_input['file_name']

    # merged_output_file_path = f"/content/downloads/{grade}/{semester}/{subject}/merged_output.pdf"
    # questions_file_path = json_input['correction_dataset']

    #@@@@@@@@@@@@@@@@@@@@@@
    #PATH For INPUT VARRIBLES CHANGE THEM ACCORDING TO YOUR SYSTEM
    merged_output_file_path = f"C:/Users/algez/Downloads/Educational_rag/{file_name}.pdf"
    questions_file_path = "C:/Users/algez/Downloads/Educational_rag/questions.csv"

    if(file_name + "_index" not in os.listdir()):
      print("NOT IN HERE")
      store_vectors(merged_output_file_path, file_name)

    # merged_output_file_path = f"/content/downloads/{grade}/{semester}/{subject}/merged_output.pdf"
    # questions_file_path = json_input['correction_dataset']
    
    if(query_type == "text"):
       
       teacher_answer = check_teacher(question, questions_file_path)

       if(teacher_answer == False):
          avaliable_tokens = check_tokens(business_remaining_number_of_tokens, end_user_remaining_tokens, question)
          if(avaliable_tokens == False):
            json_output = {"reject": 51, "type": "", "response":  {"rag_response": "", "tokens_consumed": 0}}

            with open(json_output_path, "w", encoding="utf-8") as file:
                json_final_output = json.dump(json_output, file, indent = 2, ensure_ascii=False)

            print("Insufficient Tokens")

          else:
            AI_output = rag(question, questions_file_path, file_name)
            json_output = {"reject": 403, "type": "Text", "response":  {"rag_response": AI_output, "tokens_consumed": avaliable_tokens}}
            with open(json_output_path, "w", encoding="utf-8") as file:
              json_final_output = json.dump(json_output, file, indent = 2, ensure_ascii=False)

            print("The output is from AI:")
            print(AI_output)

       else:
        json_output = {"reject": 403, "type": "text", "response":  {"rag_response": teacher_answer, "tokens_consumed": 0}}
        with open(json_output_path, "w", encoding="utf-8") as file:
          json_final_output = json.dump(json_output, file, indent = 2, ensure_ascii=False)
        print("The output is from a teacher's modified answer, No tokens are consumed:")
        print(teacher_answer)

else:
    pass