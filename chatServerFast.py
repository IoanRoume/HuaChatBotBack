import threading
import time
import os
import json
import sys
from fastapi import FastAPI, Request, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

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


def reRankingRetriever_local(query, retriever):
    retrieved_documents = retriever.invoke(query)
    documents = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in retrieved_documents]

    ranked_indexes = rerank(query, documents)
    ranked_indexes = ranked_indexes[:5]

    filtered_documents = [retrieved_documents[i] for i in ranked_indexes]
    return filtered_documents


def query_model(question, docs, chatModel, final_result_graph):

    prompt_text = rag_prompt.format(question=question, context=docs, graph=final_result_graph)
    print(prompt_text, file=sys.stderr)
    response = chatModel.create_chat_completion(messages=[{"role": "user", "content": prompt_text}])



    return response['choices'][0]['message']['content']



G = nx.Graph()

subjects = {
    "Προγραμματισμός ΙΙ": {
        "teachers": ["Γ. Παπαδόπουλος"],
        "locations": ["Αμφιθέατρο", "Εργαστήριο 4ου Ορόφου", "Εργαστήριο 2ου Ορόφου"]
    },
    "Αριθμητική Ανάλυση": {
        "teachers": ["Χ. Μιχαλακέλης", "Ε. Φιλιοπούλου"],
        "locations": ["Αμφιθέατρο"]
    },
    "Αντικειμενοστρεφής Προγραμματισμός Ι": {
        "teachers": ["Κ. Μπαρδάκη", "Α. Χαραλαμπίδης", "Β. Ευθυμίου"],
        "locations": ["Αμφιθέατρο", "Εργαστήριο 4ου Ορόφου"]
    },
    "Πιθανότητες": {
        "teachers": ["Μ. Βαμβακάρη"],
        "locations": ["Αμφιθέατρο"]
    },
    "Αρχιτεκτονική Υπολογιστών": {
        "teachers": ["Εξωτερικός Διδάσκοντες"],
        "locations": ["Αμφιθέατρο", "Εργαστήριο 4ου Ορόφου", "Εργαστήριο 2ου Ορόφου"]
    },
    "Μεθοδολογία Επιστημονικής Έρευνας": {
        "teachers": ["Χ. Σοφιανοπούλου", "A. Γασπαρινάτου"],
        "locations": ["Αμφιθέατρο", "Εργαστήριο 2ου Ορόφου"]
    },
    "Σήματα & Συστήματα": {
        "teachers": ["Π. Ριζομυλιώτης"],
        "locations": ["Αμφιθέατρο"]
    },
    "Βάσεις Δεδομένων": {
        "teachers": ["Η. Βαρλάμης", "Β. Ευθυμίου"],
        "locations": ["Εργαστήριο 2ου Ορόφου", "Αμφιθέατρο"]
    },
    "Τεχνολογίες Εφαρμογών Ιστού": {
        "teachers": ["Εξωτερικός Διδάσκων"],
        "locations": ["Αμφιθέατρο", "Εργαστήριο 4ου Ορόφου"]
    },
    "Ανάλυση Συστημάτων και Τεχνολογία Λογισμικού": {
        "teachers": ["Κ. Μπαρδάκη"],
        "locations": ["Αμφιθέατρο"]
    },
    "Υπηρεσίες και Συστήματα Διαδικτύου": {
        "teachers": ["Γ. Κουσιουρής"],
        "locations": ["Αίθουσα 2.3", "Εργαστήριο 4ου Ορόφου"]
    },
    "Ψηφιακή Επεξεργασία Εικόνας και Εφαρμογές": {
        "teachers": ["Γ. Παπαδόπουλος"],
        "locations": ["Αίθουσα 2.3", "Εργαστήριο 4ου Ορόφου"]
    },
    "Συστήματα Λήψης Αποφάσεων": {
        "teachers": ["Γ. Δέδε"],
        "locations": ["Αίθουσα 2.3", "Εργαστήριο 2ου Ορόφου"]
    },
    "Καινοτομία και Επιχειρηματικότητα": {
        "teachers": ["Στ. Λουνής"],
        "locations": ["Αίθουσα 1.2 Χαροκόπου 89"]
    },
    "Βασικές Έννοιες κι Εργαλεία DevOps": {
        "teachers": ["Α. Τσαδήμας"],
        "locations": ["Εργαστήριο 4ου Ορόφου"]
    },
    "Αλγόριθμοι και Πολυπλοκότητα)": {
        "teachers": ["Δ. Μιχαήλ"],
        "locations": ["Αμφιθέατρο"]
    },
    "Προσομοίωση": {
        "teachers": ["Β. Δαλάκας"],
        "locations": ["Αμφιθέατρο", "Εργαστήριο 2ου Ορόφου"]
    },
    "Εφαρμογές Ηλεκτρονικής και Διαδίκτυο των Πραγμάτων": {
        "teachers": ["Θ. Καμαλάκης"],
        "locations": ["Αίθουσα 2.3"]
    },
    "Οπτικές Επικοινωνίες": {
        "teachers": ["Θ. Καμαλάκης"],
        "locations": ["Αίθουσα 2.3"]
    },
    "Τεχνητή Νοημοσύνη": {
        "teachers": ["Χ. Δίου"],
        "locations": ["Αμφιθέατρο"]
    },
    "Μεταγλωττιστές": {
        "teachers": ["Α. Χαραλαμπίδης"],
        "locations": ["Αίθουσα 3.9", "Εργαστήριο 2ου Ορόφου"]
    },
    "Διδακτική της Πληροφορικής": {
        "teachers": ["Χ. Σοφιανοπούλου", "Α. Γασπαρινάτου"],
        "locations": ["Αίθουσα 3.9", "Εργαστήριο 2ου Ορόφου"]
    },
    "Παιδαγωγική Ψυχολογία": {
        "teachers": ["Δ. Ζμπάινος"],
        "locations": ["Αίθουσα 3.9"]
    },
    "Διοίκηση Έργων Πληροφορικής": {
        "teachers": ["Μ. Σταμάτη", "Ε. Φιλιοπούλου"],
        "locations": ["Αίθουσα 3.9", "Εργαστήριο 2ου Ορόφου"]
    },
    "Πληροφοριακά Συστήματα και Ηλεκτρονικό Επιχειρείν": {
        "teachers": ["Μ. Σταμάτη"],
        "locations": ["Αίθουσα 3.9"]
    },
    "Προγραμματισμός Συστημάτων": {
        "teachers": ["Εξωτερικός Διδάσκων"],
        "locations": ["Αίθουσα 3.9"]
    },
    "Κρυπτογραφία": {
        "teachers": ["Π. Ριζομυλιώτης"],
        "locations": ["Αίθουσα 3.9"]
    },
    "Διαχείριση Υπολογιστικού Νέφους": {
        "teachers": ["Εξωτερικός Διδάσκων"],
        "locations": ["Αίθουσα 3.9", "Εργαστήριο 4ου Ορόφου"]
    },
    "Διαχείριση Δικτύων Βασισμένων στο Λογισμικό": {
        "teachers": ["Ε. Λιώτου"],
        "locations": ["Εργαστήριο 2ου Ορόφου"]
    },
    "Ανάκτηση Πληροφορίας και Επεξεργασία Φυσικής Γλώσσας": {
        "teachers": ["Η. Βαρλάμης"],
        "locations": ["Εργαστήριο 2ου Ορόφου"]
    },
    "Προγραμματισμός Ι": {
        "teachers": ["Χ. Δίου", "Γ. Παπαδόπουλος", "Α. Γασπαρηνάτου"],
        "locations": ["Εργαστήριο 4ου Ορόφου","Αμφιθέατρο"]
    },
    "Διακριτά Μαθηματικά": {
        "teachers": ["Μ. Βαμβακάρη"],
        "locations": ["Αμφιθέατρο"]
    },
    "Λογική Σχεδίαση": {
        "teachers": ["Γ. Φραγκιαδάκης"],
        "locations": ["Εργαστήριο 2ου Ορόφου","Αμφιθέατρο"]
    },
    "Ψηφιακή Τεχνολογία και Εφαρμογές Τηλεματικής": {
        "teachers": ["Α. Γασπαρηνάτου"],
        "locations": ["Αμφιθέατρο"]
    },
    "Υπολογιστικά Μαθηματικά": {
        "teachers": ["Χ. Μιχαλακέλης"],
        "locations": ["Αμφιθέατρο"]
    },
    "Ψηφιακή Τεχνολογία και Εφαρμογές Τηλεματικής": {
        "teachers": ["Α. Γασπαρηνάτου"],
        "locations": ["Αμφιθέατρο"],
        "days": ["Πέμπτη"]
    },
    "Υπολογιστικά Μαθηματικά": {
        "teachers": ["Χ. Μιχαλακέλης"],
        "locations": ["Αμφιθέατρο"],
        "days": ["Παρασκευή"]
    },
    "Αντικειμενοστραφής Προγραμματισμός ΙΙ": {
        "teachers": ["Α. Χαραλαμπίδης"],
        "locations": ["Αμφιθέατρο"],
        "days": ["Δευτέρα"]
    },
    "Στατιστική": {
        "teachers": ["Μ. Βαμβακάρη"],
        "locations": ["Αμφιθέατρο", "Εργαστήριο 2ου Ορόφου"],
        "days": ["Τρίτη"]
    },
    "Δομές Δεδομένων": {
        "teachers": ["Δ. Μιχαήλ"],
        "locations": ["Αμφιθέατρο", "Εργαστήριο 2ου Ορόφου"],
        "days": ["Τρίτη", "Πέμπτη"]
    },
    "Λειτουργικά Συστήματα": {
        "teachers": ["Α. Τσαδήμας", "Γ. Κουσιουρής"],
        "locations": ["Εργαστήριο 4ου Ορόφου", "Αμφιθέατρο"],
        "days": ["Δευτέρα", "Παρασκευή"]
    },
    "Δίκτυα Υπολογιστών": {
        "teachers": ["Ε. Λιώτου"],
        "locations": ["Εργαστήριο 2ου Ορόφου", "Αμφιθέατρο"],
        "days": ["Δευτέρα", "Πέμπτη"]
    },
    "Κατανεμημένα Συστήματα": {
        "teachers": ["Α. Τσαδήμας", "Μ. Νικολαϊδη"],
        "locations": ["Εργαστήριο 2ου Ορόφου", "Αμφιθέατρο"],
        "days": ["Δευτέρα", "Τετάρτη"]
    },
    "Πληροφοριακά Συστήματα": {
        "teachers": ["Μ. Σταμάτη"],
        "locations": ["Αμφιθέατρο"],
        "days": ["Δευτέρα"]
    },
    "Τεχνολογίες Διαδικτύου": {
        "teachers": ["Ε. Λιώτου"],
        "locations": ["Αίθουσα 3.7"],
        "days": ["Τρίτη"]
    },
    "Οικονομικά της Ψηφιακής Τεχνολογίας": {
        "teachers": ["Χ. Μιχαλακέλης"],
        "locations": ["Αίθουσα 2.3"],
        "days": ["Τρίτη"]
    },
    "Τηλεπικοινωνιακά Συστήματα": {
        "teachers": ["Θ. Καμαλάκης"],
        "locations": ["Αμφιθέατρο"],
        "days": ["Τετάρτη"]
    },
    "Προηγμένα Θέματα Λειτουργικών Συστημάτων": {
        "teachers": ["Χ. Ανδρίκος"],
        "locations": ["Αίθουσα 3.9"],
        "days": ["Τετάρτη"]
    },
    "Σχεδίαση ΒΔ και Κατανεμημένες ΒΔ": {
        "teachers": ["Β. Ευθυμίου"],
        "locations": ["Αίθουσα 2.3", "Εργαστήριο 2ου Ορόφου"],
        "days": ["Παρασκευή"]
    },
    "Διαχείριση Επιχειρηματικών Διαδικασιών στην Εφοδιαστική Αλυσίδα": {
        "teachers": ["Κ. Μπαρδάκη"],
        "locations": ["Αίθουσα 2.3"],
        "days": ["Παρασκευή"]
    },
    "Ανάπτυξη Κινητών Εφαρμογών": {
        "teachers": ["Ε. Χονδρογιάννης"],
        "locations": ["Αίθουσα 2.3"],
        "days": ["Παρασκευή"]
    },
    "Απόδοση Συστημάτων": {
        "teachers": ["Γ. Κουσιουρής"],
        "locations": ["Αίθουσα 3.9", "Εργαστήριο 2ου Ορόφου"],
        "days": ["Δευτέρα"]
    },
    "Κοινωνία και ΤΠΕ": {
        "teachers": ["Χ. Σοφιανοπούλου"],
        "locations": ["Αίθουσα 3.9"],
        "days": ["Δευτέρα"]
    },
    "Εφαρμογές Τηλεματικής στις Μεταφορές και την Υγεία": {
        "teachers": ["Γ. Δημητρακόπουλος", "Χ. Δίου"],
        "locations": ["Αίθουσα 3.9"],
        "days": ["Δευτέρα"]
    },
    "Συστήματα Κινητών Επικοινωνιών": {
        "teachers": ["Γ. Δημητρακόπουλος"],
        "locations": ["Αίθουσα 3.9"],
        "days": ["Τρίτη"]
    },
    "Αποτίμηση Επενδύσεων ΤΠΕ": {
        "teachers": ["Χ. Μιχαλακέλης"],
        "locations": ["Αίθουσα 3.9"],
        "days": ["Τρίτη"]
    },
    "Πληροφορική και Εκπαίδευση": {
        "teachers": ["Α. Γασπαρινάτου"],
        "locations": ["Αίθουσα 3.9", "Εργαστήριο 4ου Ορόφου"],
        "days": ["Τρίτη"]
    },
    "Μηχανική Μάθηση και Εφαρμογές": {
        "teachers": ["Χ. Δίου"],
        "locations": ["Αίθουσα 3.9"],
        "days": ["Τετάρτη"]
    },
    "Αξιολόγηση Συστημάτων και Διεπαφών": {
        "teachers": ["Γ. Δέδε"],
        "locations": ["Αμφιθέατρο", "Εργαστήριο 2ου Ορόφου"],
        "days": ["Τετάρτη"]
    },
    "Διδακτική ρομποτικής και εκπαίδευση STEM": {
        "teachers": ["Ε. Φιλιοπούλου"],
        "locations": ["Αίθουσα 3.9", "Εργαστήριο 4ου Ορόφου"],
        "days": ["Πέμπτη"]
    },
    "Παράλληλοι Υπολογιστές και Αλγόριθμοι": {
        "teachers": ["Π. Μιχαήλ"],
        "locations": ["Εργαστήριο 4ου Ορόφου"],
        "days": ["Πέμπτη"]
    },
    "Τεχνολογία Γραφημάτων και Εφαρμογές": {
        "teachers": ["Δ. Μιχαήλ"],
        "locations": ["Αίθουσα 3.9", "Εργαστήριο 2ου Ορόφου"],
        "days": ["Πέμπτη"]
    },
    "Εξόρυξη Δεδομένων": {
        "teachers": ["Η. Βαρλάμης"],
        "locations": ["Αίθουσα 3.9", "Εργαστήριο 2ου Ορόφου"],
        "days": ["Παρασκευή"]
    },
    "Ασφάλεια Πληροφοριακών Συστημάτων": {
        "teachers": ["Π. Ριζομυλιώτης"],
        "locations": ["Αμφιθέατρο"],
        "days": ["Παρασκευή"]
    }
    
}


for subject, data in subjects.items():
    G.add_node(subject, type="subject")

    for teacher in data["teachers"]:
        G.add_node(teacher, type="teacher")
        G.add_edge(subject, teacher) 

    for location in data["locations"]:
        G.add_node(location, type="location")
        G.add_edge(subject, location)  


with open("/home/it2021087/chatBot/Hua/stopwords.txt", "r", encoding="utf-8") as f:
    stop_words = set(word.strip() for word in f.readlines())

def find_graph_results(graph, user_query):
    global final_result_graph
    normalized_query = unidecode.unidecode(user_query).lower()
    result= ""
    nodes = [n for n, d in graph.nodes(data=True)]

    matched_nodes = [t for t in nodes if normalized_query in unidecode.unidecode(t).lower()]

    if matched_nodes:
        for node in matched_nodes:
            neighbors = [n for n in graph.neighbors(node)]
            if graph.nodes[node]['type'] == 'teacher':
              result = f"Ο διδάσκοντας {node} έχει τα μαθήματα: {', '.join(neighbors)}\n"
            elif graph.nodes[node]['type'] == 'subject':
              teachers = []
              rooms = []
              
              for neighbor in neighbors:
                  neighbor_type = graph.nodes[neighbor].get('type', 'unknown') 
                  
                  if neighbor_type == 'teacher':
                      teachers.append(neighbor)
                  elif neighbor_type == 'location':
                      rooms.append(neighbor)
              
              teacher_text = f"Διδάσκεται από: {', '.join(teachers)}" if teachers else ""
              room_text = f"Στην αίθουσα: {', '.join(rooms)}" if rooms else ""
              
              result = f"Το μάθημα {node} {teacher_text} {room_text}".strip() + "\n"

            else:
              result = f"Η Αίθουσα {node} έχει τα μαθήματα: {', '.join(neighbors)}\n"
            final_result_graph.add(result)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# last_request_time = time.time()
# inactive_time = 3600
# allowCleanup = True
# areModelsWiped = False
# lock = threading.Lock()

# def check_inactivity():
#     """Background thread to check for inactivity."""
#     global last_request_time
#     while True:
#         with lock:
#             elapsed_time = time.time() - last_request_time
#             if elapsed_time > inactive_time and allowCleanup:
#                 print("No activity for 1 Hour. Cleaning Cache...", file=sys.stderr)
#                 cleanup()
#         time.sleep(900)


# def cleanup():
#     global areModelsWiped, tokenizer, reranker_model, retriever, vectordb, embedding, allowCleanup, chatModel
#     torch.cuda.empty_cache()
#     gc.collect()
#     allowCleanup = False
#     del tokenizer, reranker_model, retriever, vectordb, embedding, chatModel
#     areModelsWiped = True

# @app.middleware("http")
# async def update_last_request_time(request: Request, call_next):
#     global last_request_time, allowCleanup, areModelsWiped
#     # allowCleanup = True
#     last_request_time = time.time()
#     return await call_next(request)


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


# def load_models():
#     global areModelsWiped, tokenizer,reranker_model, retriever, vectordb, embedding, chatModel
#     with lock:
#         if not areModelsWiped:
#             return
        
#         print(f"Reinitializing models...", file=sys.stderr)

#         tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True)       
#         reranker_model = AutoModelForSequenceClassification.from_pretrained("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")

#         persist_directory = 'db_el'
#         embedding = HuggingFaceEmbeddings(
#             model_name="nomic-ai/nomic-embed-text-v2-moe",
#             model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'trust_remote_code': True},
#             encode_kwargs={'normalize_embeddings': False}
#         )

#         vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
#         retriever = vectordb.as_retriever(search_kwargs={"k": 40})

#         chatModel = Llama.from_pretrained(repo_id="bartowski/ilsp_Llama-Krikri-8B-Instruct-GGUF", filename="ilsp_Llama-Krikri-8B-Instruct-Q4_K_M.gguf", n_ctx=8192, n_gpu_layers=-1)

#         areModelsWiped = False
#         print("Models reinitialized", file=sys.stderr)


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def run_chat(request: ChatRequest):
    global final_result_graph
    #global chatModel
    # load_models()
    final_result_graph = set()
    try:
        words = re.findall(r'\b\w+\b', request.message.lower()) 
        filtered_words = [word for word in words if word not in stop_words]
        for word in filtered_words:
            find_graph_results(G, word)
        print(final_result_graph)
        #chatModel = Llama.from_pretrained(repo_id=repo_id_chat_model, filename=filename_chat_model, n_ctx=8192, n_gpu_layers=-1)
        documents = reRankingRetriever_local(request.message, retriever)
        generated_answer = query_model(request.message, documents, chatModel, final_result_graph)
        #del chatModel
        #chatModel = None
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


# thread = threading.Thread(target=check_inactivity, daemon=True)
# thread.start()

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
    Είσαι ένας έξυπνος βοηθός που απαντά σε ερωτήσεις βασισμένος σε πληροφορίες από τη βάση γνώσεων του Χαροκοπείου Πανεπιστημίου. 
    Απάντησε στην παρακάτω ερώτηση με σαφήνεια και ακρίβεια χρησιμοποιώντας τα παρεχόμενα αποσπάσματα από τη βάση γνώσεων. 
    Αν δεν γνωρίζεις την απάντηση, πες ότι δεν είσαι σίγουρος αντί να δώσεις λανθασμένη πληροφορία.
    Αν η ερώτηση είναι εκτός θέματος, απαντάς ευγενικά ότι δεν διαθέτεις τις σχετικές πληροφορίες.
    Αν η ερώτηση είναι στα Αγγλικά τότε πρέπει και εσύ να απαντήσεις στα Αγγλικά, αν είναι στα Ελληνικά τότε πρέπει να απαντήσεις στα Ελληνικά.
    Δεν πρέπει να αναφέρεις ότι η απάντηση σου προέρχεται από την βάση γνώσεων.

    Ερώτηση: {question}

    Πληροφορίες από τη βάση γνώσεων:
    {graph}
    ---
    {context}

    Απάντηση:
    """

    rag_prompt = PromptTemplate(
        input_variables=["question", "context", "graph"],
        template=template
    )
    print(f"Loaded chat model")

    final_result_graph = set()


    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)