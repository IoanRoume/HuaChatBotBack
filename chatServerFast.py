import threading
import time
import os
import json
import sys
from fastapi import FastAPI, Request, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from langchain.schema import Document
from llama_cpp import Llama
from langchain.text_splitter import TextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llama_cpp import Llama
from langchain.prompts import PromptTemplate
import torch
import gc

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
import unidecode  
import re

class MarkdownTitleTextSplitter(TextSplitter):
    def split_text(self, text: str):
        sections = []
        current_section = []
        previous_line = ""
        skipLine = False

        for line in text.splitlines():
            #Check if the line starts with a Markdown header (e.g., #, ##, ###)
            if line == "":
              continue
            elif line.startswith("#"):
                current_section.append(previous_line)
                if current_section:  # Save the current section if it exists
                    sections.append("\n".join(current_section))
                current_section = [line]  # Start a new section
                skipLine = True
            #Check if the line starts with '**' (indicating bold text or a special marker)
            elif line.startswith("**"):
                current_section.append(previous_line)
                if current_section:  # Save the current section if it exists
                    sections.append("\n".join(current_section))
                current_section = [line]  # Start a new section
                skipLine = True
            elif line.startswith("|"):
                current_section.append(previous_line)
                if current_section:  # Save the current section if it exists
                    sections.append("\n".join(current_section))
                current_section = [line]  # Start a new section
                skipLine = True
            #Check if the line is a separator (a line with only '=' characters)
            elif line.startswith("==") or line.startswith("--"):

                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [previous_line]
            else:
              if skipLine == False:
                current_section.append(previous_line)
              elif skipLine:
                skipLine = False

            previous_line = line
        #Add the final section if there's remaining content
        if current_section:
            sections.append("\n".join(current_section))

        return sections



def rerank(query, documents, batch_size=16):
    pairs = [(query, doc) for doc in documents]
    ranked_indexes = []
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)

    torch.cuda.empty_cache()

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.view(-1).cpu().tolist()  #Get scores

    #Sort documents by score and store indexes
    ranked_indexes = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return ranked_indexes




def reRankingRetriever_local(query, retriever, history):
    print(f"Original Query: {query}\n\n")
    rephrase_prompt_format = rephrase_prompt.format(history=history, query=query)
    rephraseResponse = chatModel.create_chat_completion(messages=[{"role": "system", "content": rephrase_prompt_format}])

    match = re.search(r'"([^"]*)"', rephraseResponse['choices'][0]['message']['content'])
    if match:
        question = match.group(1)
    print(f"Rephrased Query: {question}\n\n")
    retrieved_documents = retriever.invoke(question)
    documents = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in retrieved_documents]

    ranked_indexes = rerank(question, documents)
    
    ranked_indexes = ranked_indexes[:5]
    filtered_documents = [retrieved_documents[i] for i in ranked_indexes]

    return filtered_documents, question



def query_model(question, docs, chatModel, history):
    formatted_docs = "\n".join([f"\n\t{i+1} - {doc.page_content}" for i, doc in enumerate(docs)])
    prompt_text = rag_prompt.format(question=question, context=formatted_docs, history=history) 
    print(prompt_text)
    response = chatModel.create_chat_completion(messages=[{"role": "user", "content": prompt_text}])
    print(response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content']



G = nx.Graph()

with open('/home/it2021087/chatBot/Hua/subjects.json', 'r', encoding='utf-8') as f:
    subjects = json.load(f)


for subject, data in subjects.items():
    G.add_node(subject, type="subject")

    for teacher in data["teachers"]:
        G.add_node(teacher, type="teacher")
        G.add_edge(subject, teacher)  

    for location in data["locations"]:
        G.add_node(location, type="location")
        G.add_edge(subject, location)  

    for day in data["days"]:
        G.add_node(day,type = "day")
        G.add_edge(subject, day)

    for semester in data["semester"]:
        G.add_node(semester,type = "semester")
        G.add_edge(subject, semester)

    for year in data["year"]:
        G.add_node(year,type = "year")
        G.add_edge(subject, year)  


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/ping")
async def ping():
    return {"message": "pong"}


FEEDBACK_FILE = "feedback.md"
ID_TRACKER_FILE = "last_id.txt"

def get_next_id():
    """Retrieve the last used ID and increment it."""
    if not os.path.exists(ID_TRACKER_FILE):
        with open(ID_TRACKER_FILE, "w") as f:
            f.write("1")
        return 1

    with open(ID_TRACKER_FILE, "r") as f:
        last_id = int(f.read().strip())

    new_id = last_id + 1
    with open(ID_TRACKER_FILE, "w") as f:
        f.write(str(new_id))

    return new_id


from typing import List, Dict
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]]

@app.post("/chat")
async def run_chat(request: ChatRequest):
    try:
        message , history = request.message, request.history
        documents, newQuestion = reRankingRetriever_local(message, retriever, history)
        generated_answer = query_model(newQuestion, documents, chatModel, history)

        torch.cuda.empty_cache()
        gc.collect()
        return {"message": generated_answer}
    except Exception as e:
        print(f"ERROR in /chat: {str(e)}", file=sys.stderr)
        return {"error": "Internal server error"}
    

@app.get("/sessionEnd")
async def session_end(feedback: str = Query(None)):
    if feedback:
        try:
            feedback_list = json.loads(feedback)
            with open("feedback.md", "a") as f:
                for feedback_entry in feedback_list:
                    nextId = get_next_id()
                    f.write(f"### ID: {nextId}\n")
                    f.write(f"**Question: ** {feedback_entry.get('question', 'Unknown')}\n")
                    f.write(f"**Answer: ** {feedback_entry.get('answer', 'Unknown')}\n")
                    f.write(f"**Feedback: ** {'GOOD' if feedback_entry.get('feedback') == 'up' else 'BAD'}\n")
                    f.write(f"**Comment: ** {feedback_entry.get('comment', 'No comment')}\n")
                    f.write(f"**Time: ** {datetime.now()}\n\n")
        except json.JSONDecodeError:
            return {"message": "Invalid feedback format"}, 400
    return {"message": "Session data received"}
    


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str
    comment: str

@app.post("/feedback")
async def save_feedback(feedback: FeedbackRequest):
    user_id = get_next_id()
    feedback_entry = {
        "id": user_id,
        "question": feedback.question,
        "answer": feedback.answer,
        "feedback": "GOOD" if feedback.feedback == "up" else "BAD",
        "comment": feedback.comment,
    }
    try:
        with open(FEEDBACK_FILE, "a") as f:
            f.write(f"### ID: {feedback_entry['id']}\n")
            f.write(f"**Question: ** {feedback_entry['question']}\n")
            f.write(f"**Answer: ** {feedback_entry['answer']}\n")
            f.write(f"**Feedback: ** {feedback_entry['feedback']}\n")
            f.write(f"**Comment: ** {feedback_entry['comment']}\n")
            f.write(f"**Time: ** {datetime.now()}\n\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")
    return {"message": "Feedback saved!", "id": user_id}



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    directory_path = '/home/it2021087/chatBot/Hua/scraped_pages_el'

    loader = DirectoryLoader(directory_path, glob="*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    #Apply the custom Markdown splitter
    markdown_splitter = MarkdownTitleTextSplitter()
    texts = markdown_splitter.split_documents(documents)

    #Persist embeddings with Chroma
    persist_directory = 'db_el'
    print(f"Loading embeding model")
    model_name = "nomic-ai/nomic-embed-text-v2-moe"
    model_kwargs = {'device': device, 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)


    #Process Graph and add information to the VectorStore
    graph_nodes = [n for n in G.nodes()]
    graph_results = []
    for node in graph_nodes:
        neighbors = list(G.neighbors(node))
        node_type = G.nodes[node].get('type', 'unknown')
        if node_type == 'teacher':
            result = f"Ο διδάσκοντας {node} έχει τα μαθήματα: {', '.join(neighbors)}\n"

        elif node_type == 'subject':
            teachers = [n for n in neighbors if G.nodes[n].get('type') == 'teacher']
            rooms = [n for n in neighbors if G.nodes[n].get('type') == 'location']
            days = [n for n in neighbors if G.nodes[n].get('type') == 'day']
            semesters = [n for n in neighbors if G.nodes[n].get('type') == 'semester']
            years = [n for n in neighbors if G.nodes[n].get('type') == 'year']

            teacher_text = f"Διδάσκεται από: {', '.join(teachers)}" if teachers else ""
            room_text = f"Στην αίθουσα: {', '.join(rooms)}" if rooms else ""
            days_text = f"Στις ημέρες: {', '.join(days)}" if days else ""
            semester_text = f"Στο {', '.join(semesters)}" if semesters else ""
            year_text = f"Στο {', '.join(years)}" if years else ""

            result = f"Το μάθημα {node}, {teacher_text}, {room_text}, {days_text}, {semester_text}, {year_text}".strip() + "\n"

        elif node_type == 'location':
            result = f"Η Αίθουσα {node} έχει τα μαθήματα: {', '.join(neighbors)}\n"

        elif node_type == 'day':
            result = f"Στην ημέρα {node} διεξάγονται τα μαθήματα: {', '.join(neighbors)}\n"

        else:
            result = f"To {node} περιλαμβάνει τα μαθήματα: {', '.join(neighbors)}\n"

        graph_results.append(result)

    graph_documents = [Document(page_content=result) for result in graph_results]
    vectordb.add_documents(graph_documents)
    retriever = vectordb.as_retriever(search_kwargs={"k": 40})  


                                                                     




    print(f"Loading reranking model")
    model_name = "Alibaba-NLP/gte-multilingual-reranker-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True).to(device)
    print(f"Loaded reranking model")

    print(f"Loading chat model")
    repo_id_chat_model = "bartowski/ilsp_Llama-Krikri-8B-Instruct-GGUF"

    filename_chat_model = "ilsp_Llama-Krikri-8B-Instruct-Q4_K_M.gguf"

    chatModel = Llama.from_pretrained(repo_id=repo_id_chat_model, filename=filename_chat_model, n_ctx=8192, n_gpu_layers=-1)  
    #chatModel = None

    template = """
    Είσαι ένας έξυπνος βοηθός που απαντά σε ερωτήσεις βασισμένος σε πληροφορίες από τη βάση γνώσεων του Χαροκοπείου Πανεπιστημίου, τμήμα Πληροφορικής Και Τηλεματικής. 
    Απάντησε στην παρακάτω ερώτηση με σαφήνεια και ακρίβεια χρησιμοποιώντας τα παρεχόμενα αποσπάσματα από τη βάση γνώσεων. 
    Αν δεν γνωρίζεις την απάντηση, πες ότι δεν είσαι σίγουρος αντί να δώσεις λανθασμένη πληροφορία.
    Αν η ερώτηση είναι εκτός θέματος, απαντάς ευγενικά ότι δεν διαθέτεις τις σχετικές πληροφορίες.
    Αν η ερώτηση είναι στα Αγγλικά τότε πρέπει και εσύ να απαντήσεις στα Αγγλικά, αν είναι στα Ελληνικά τότε πρέπει να απαντήσεις στα Ελληνικά.
    Πρέπει να διαβάζεις προσεκτικά το περιεχόμενο της ερώτησης για να καταλάβεις τι ζητείται.
    Δεν πρέπει να δίνεις πληροφορίες που δεν ζητήθηκαν.

    Ιστορικό Συνομιλίας: {history}

    Νέα Ερώτηση: {question}

    Πληροφορίες από τη βάση γνώσεων:
{context}

    Απάντηση:
    """

    rag_prompt = PromptTemplate(
        input_variables=["question", "context", "history"],
        template=template
    )


    template_rephrase = """
Δεδομένου ενός ιστορικού συνομιλίας και της τελευταίας ερώτησης του χρήστη,  
η οποία μπορεί να αναφέρεται σε προηγούμενο περιεχόμενο της συνομιλίας,  
διαμόρφωσε μια αυτόνομη ερώτηση που να μπορεί να γίνει κατανοητή χωρίς το ιστορικό.  

# **Οδηγίες**  
- **ΜΗΝ απαντήσεις στην ερώτηση.**  
- **Επέστρεψε μόνο την αναδιατυπωμένη ερώτηση μέσα σε εισαγωγικά ("...").**  
- **Αν η τελευταία ερώτηση δεν χρειάζεται αλλαγή, απλώς επέστρεψέ την όπως είναι, μέσα σε εισαγωγικά.**  
- **Αν η τελευταία ερώτηση δεν σχετίζεται με το ιστορικό, μην την αλλάζεις.**  
- **Μην προσθέτεις επιπλέον πληροφορίες, σχόλια ή απαντήσεις.**  

## **Παραδείγματα**  
### Παράδειγμα 1  
- **Ιστορικό:** "Ποιος είναι ο A;"  
- **Τελευταία ερώτηση:** "Πού βρίσκεται το γραφείο του;"  
- **Αναμενόμενη έξοδος:** `"Πού βρίσκεται το γραφείο του A;"`  

### Παράδειγμα 2  
- **Ιστορικό:** (Κενό)  
- **Τελευταία ερώτηση:** "Ποια μαθήματα διδάσκει ο B;"  
- **Αναμενόμενη έξοδος:** `"Ποια μαθήματα διδάσκει ο B;"`  

### Παράδειγμα 3  
- **Ιστορικό:** "Ποιος είναι ο A;" 
- **Τελευταία ερώτηση:** "Ποια είναι τα μαθήματα του Τέταρτου Έτους;"  
- **Αναμενόμενη έξοδος:** `"Ποια είναι τα μαθήματα του Τέταρτου Έτους;"`  

Ιστορικό Συνομιλίας: {history}  
Τελευταία ερώτηση: {query}  

Επαναδιατυπωμένο Κείμενο:  
"""

    rephrase_prompt = PromptTemplate(
        input_variables=["history", "query"],
        template=template_rephrase
    )


    print(f"Loaded chat model")



    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)