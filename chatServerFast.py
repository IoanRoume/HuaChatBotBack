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

def make_documents_fromGraph(graph_nodes):
        global graph_results
        for node in graph_nodes:
            neighbors = list(G.neighbors(node))
            node_type = G.nodes[node].get('type', 'unknown')
            if node_type == 'teacher':
                result = f"Ο διδάσκοντας {node} έχει τα μαθήματα: -" + '\n-  '.join([
                f"{neighbor} ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'year']}), ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'semester']}) - {[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'decision']}"
                for neighbor in neighbors if G.nodes[neighbor].get('type', 'unknown') == 'subject'
                ]) + "\n"

            elif node_type == 'subject':
                teachers = [n for n in neighbors if G.nodes[n].get('type') == 'teacher']
                rooms = [n for n in neighbors if G.nodes[n].get('type') == 'location']
                days = [n for n in neighbors if G.nodes[n].get('type') == 'day']
                semesters = [n for n in neighbors if G.nodes[n].get('type') == 'semester']
                years = [n for n in neighbors if G.nodes[n].get('type') == 'year']
                decision = [n for n in neighbors if G.nodes[n].get('type') == 'decision']

                teacher_text = f"Διδάσκεται από: {', '.join(teachers)}" if teachers else ""
                room_text = f"Στην αίθουσα: {', '.join(rooms)}" if rooms else ""
                days_text = f"Στις ημέρες: {', '.join(days)}" if days else ""
                semester_text = f"Στο {', '.join(semesters)}" if semesters else ""
                year_text = f"Στο {', '.join(years)}" if years else ""
                decision_text = f"Το μάθημα είναι : {''.join(decision)}"

                result = f"Το μάθημα {node}, {teacher_text}, {room_text}, {days_text}, {semester_text}, {year_text}, {decision_text}".strip() + "\n"

            elif node_type == 'location':
                result = f"Η Αίθουσα {node} έχει τα μαθήματα: -"+ '\n-  '.join([
                f"{neighbor} ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'year']}), ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'semester']}) - {[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'decision']}"
                for neighbor in neighbors if G.nodes[neighbor].get('type', 'unknown') == 'subject'
                ]) + "\n"

            elif node_type == 'day':
                result = f"Στην ημέρα {node} διεξάγονται τα μαθήματα: -"+'\n-  '.join([
                f"{neighbor} ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'year']}), ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'semester']}) - {[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'decision']}"
                for neighbor in neighbors if G.nodes[neighbor].get('type', 'unknown') == 'subject'
                ]) + "\n"
            elif node_type == 'faculty':
                result = f"Οι {node} του τμήματος Πληροφορικής και τηλεματικής είναι: {', '.join(neighbors)}\n"
            elif node_type == 'career_op':
                result = f"Για να εξελιχθείς, να ειδικευτείς ή και να ασχοληθείς ως {node}, τα κατάλληλα μαθήματα που θα σε προετοιμάσουν καλύτερα είναι: -"+'\n-  '.join([
                f"{neighbor} ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'year']}), ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'semester']}) - {[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'decision']}"
                for neighbor in neighbors if G.nodes[neighbor].get('type', 'unknown') == 'subject'
                ]) + "\n"
            elif node_type == 'decision':
                result = f"Τα μαθήματα που είναι {node}, στο τμήμα Πληροφορικής και Τηλεματικής είναι: -"+'\n-  '.join([
                f"{neighbor} ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'year']}), ({[n for n in list(G.neighbors(neighbor)) if G.nodes[n].get('type', 'unknown') == 'semester']})"
                for neighbor in neighbors if G.nodes[neighbor].get('type', 'unknown') == 'subject'
                ]) + "\n"
            elif node_type == 'secretary':
                staff = []
                email = []
                phone = []
                for neightbor in neighbors:
                    neighbor_type = G.nodes[neightbor]['type']

                    if neighbor_type == "secretary_staff":
                        staff.append(neightbor)
                    if neighbor_type == "secretary_email":
                        email.append(neightbor)
                    if neighbor_type == "secretary_phone":
                        phone.append(neightbor)
                    staff_text = f"{', '.join(staff)}"

                result = f"Το προσωπικό της {node} του τμήματος Πληροφορικής και τηλεματικής είναι: {', '.join(staff)}\n"
                graph_results.append(result)
                result = f"Το Εmail της {node} του τμήματος Πληροφορικής και τηλεματικής είναι: {', '.join(email)}, και το Τηλέφωνο της {node} είναι: {', '.join(phone)}\n"

            else:
                result = f"To {node} περιλαμβάνει τα μαθήματα: {', '.join(neighbors)}\n"

            graph_results.append(result)

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
        query = match.group(1)
    print(f"Rephrased Query: {query}\n\n")
    retrieved_documents = retriever.invoke(query)
    documents = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in retrieved_documents]

    ranked_indexes = rerank(query, documents)
    
    ranked_indexes = ranked_indexes[:5]
    filtered_documents = [retrieved_documents[i] for i in ranked_indexes]

    return filtered_documents, query



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

with open('/home/it2021087/chatBot/Hua/staff.json', 'r', encoding='utf-8') as f:
    staff = json.load(f)

G.add_node("Διδάσκοντες", type = "faculty")

for staff, data in staff.items():
  G.add_node(staff,type = "secretary")

  for staff_name in data["Γραμματέας"]:
    G.add_node(staff_name, type="secretary_staff")
    G.add_edge(staff,staff_name)
  for email in data["email"]:
    G.add_node(email, type="secretary_email")
    G.add_edge(staff,email)
  for phone in data["Τηλέφωνο"]:
    G.add_node(phone, type="secretary_phone")
    G.add_edge(staff,phone)

for subject, data in subjects.items():
    G.add_node(subject, type="subject")

    for teacher in data["teachers"]:
        G.add_node(teacher, type="teacher")
        G.add_edge(subject, teacher)  
        G.add_edge("Διδάσκοντες",teacher)

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
        
    for career in data["career_op"]:
      G.add_node(career,type = "career_op")
      G.add_edge(subject, career)

    for decision in data["decision"]:
      G.add_node(decision,type = "decision")
      G.add_edge(subject, decision)


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

def format_history(chat_history):

    if len(chat_history) < 2:
        return ""  #Αν δεν υπάρχει προηγούμενη απάντηση, δεν προσθέτουμε ιστορικό

    last_user_question = None
    last_bot_answer = None

    #Διατρέχουμε το ιστορικό από το τέλος προς την αρχή
    for entry in reversed(chat_history):
        if entry["role"] == "bot" and last_bot_answer is None:
            last_bot_answer = entry["content"]
        elif entry["role"] == "user" and last_user_question is None:
            last_user_question = entry["content"]
            break  #Μόλις βρούμε και τα δύο, σταματάμε

    if last_user_question and last_bot_answer:
        return f"- Προηγούμενη Ερώτηση από χρήστη: {last_user_question}\n\n- Προηγούμενη Απάντησή σου: {last_bot_answer}"

    return ""


from typing import List, Dict
class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]]

@app.post("/chat")
async def run_chat(request: ChatRequest):
    try:
        message , history = request.message, request.history
        history = format_history(history)
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
    make_documents_fromGraph(graph_nodes)
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

    filename_chat_model = "ilsp_Llama-Krikri-8B-Instruct-Q6_K.gguf"

    chatModel = Llama.from_pretrained(repo_id=repo_id_chat_model, filename=filename_chat_model, n_ctx=8192, n_gpu_layers=-1)  

    template = """
Είσαι ένας έξυπνος βοηθός που απαντά σε ερωτήσεις βασισμένος **μόνο** σε πληροφορίες από τη βάση γνώσεων του Χαροκοπείου Πανεπιστημίου, Τμήμα Πληροφορικής και Τηλεματικής.

## **Κανόνες απάντησης**:
- **Απαγορεύεται** να απαντάς βασιζόμενος αποκλειστικά στο ιστορικό συνομιλίας.
- Το ιστορικό συνομιλίας υπάρχει **μόνο** για να σε βοηθήσει να κατανοήσεις το πλαίσιο της νέας ερώτησης, **όχι** για να αντλήσεις απαντήσεις από αυτό.
- Πρέπει να απαντάς **αποκλειστικά** χρησιμοποιώντας τις παρεχόμενες πληροφορίες από τη βάση γνώσεων.
- Αν η βάση γνώσεων δεν περιέχει την απάντηση, απαντάς: **"Δεν είμαι σίγουρος. Για περισσότερες πληροφορίες, μπορείτε να επικοινωνήσετε με τη γραμματεία του τμήματος."**  
- Αν η ερώτηση είναι εκτός θέματος, απαντάς: **"Δεν διαθέτω πληροφορίες για αυτό το ζήτημα."**  
- Αν η ερώτηση είναι στα Αγγλικά, απαντάς στα Αγγλικά. Αν είναι στα Ελληνικά, απαντάς στα Ελληνικά.
- Πρέπει να διαβάζεις **προσεκτικά** το περιεχόμενο της ερώτησης ώστε να απαντήσεις **ακριβώς** σε αυτό που ζητείται, χωρίς περιττές πληροφορίες ή αυθαίρετες ερμηνείες.
- Αν η ερώτηση δεν αναφέρεται σε πληροφορίες της βάσης γνώσεων, πρέπει να παραμείνεις αμέτοχος και να μην προσπαθήσεις να δώσεις δική σου εκτίμηση ή εικασία.
- **Καμία υπόθεση δεν επιτρέπεται.** Αν δεν υπάρχει επαρκές περιεχόμενο στην βάση γνώσεων για να απαντήσεις, πρέπει να παραμείνεις ακριβής και να πεις ότι δεν έχεις την πληροφορία.
- **Δεν πρέπει να απαντάς με τρόπο που υποδηλώνει ότι διαβάζεις ή επεξεργάζεσαι περιεχόμενο από την βάση γνώσεων.** Οι απαντήσεις πρέπει να φαίνονται φυσικές και να μην δημιουργούν την εντύπωση ότι ανακτάς ή αντιγράφεις ακριβώς πληροφορίες από μια βάση δεδομένων ή άλλο έγγραφο.
- Όταν αναφέρεσαι σε πληροφορίες, **πρέπει να τις ενσωματώνεις φυσικά** χωρίς να φανερώνεις την προέλευση της πληροφορίας, ώστε να μη δημιουργείται η αίσθηση ότι προέρχεται από εξωτερική πηγή.

## **Παραδείγματα για κατανόηση**:
    **Παράδειγμα 1:**  
    - **Ιστορικό:** "Ποιος είναι ο καθηγητής του μαθήματος Α;"
    - **Νέα Ερώτηση:** "Ποια μαθήματα περιλαμβάνει το 1ο Έτος;"
    - **Πληροφορίες από τη βάση γνώσεων:** "To Πρώτο Έτος περιλαμβάνει τα μαθήματα: Προγραμματισμός ΙΙ, Αριθμητική Ανάλυση, Αντικειμενοστρεφής Προγραμματισμός Ι, Πιθανότητες, Αρχιτεκτονική Υπολογιστών, Προγραμματισμός Ι, Διακριτά Μαθηματικά, Λογική Σχεδίαση, Ψηφιακή Τεχνολογία και Εφαρμογές Τηλεματικής, Υπολογιστικά Μαθηματικά"
    - **Αναμενόμενη απάντηση:**  
        "Τα μαθήματα του 1ου Έτους περιλαμβάνουν: [λίστα μαθημάτων]."

     **Παράδειγμα 2:**  
    - **Ιστορικό:** (κενό)
    - **Νέα Ερώτηση:** "Ποια μαθήματα έχει Ο ο καθηγητής Α;"
    - **Πληροφορίες από τη βάση γνώσεων:** "1- Μάθημα1 επιλογής διδάσκεται από τον Α, 2- Μάθημα2 υποχρεωτικό διδάσκεται από τον Α, 3- Ο καθηγητής Α διδάσκει το μάθημα1,μάθημα2, μάθημα3"
    - **Αναμενόμενη απάντηση:**  
        "Ο Α διδάσκει τα μαθήματα: Μάθημα1 (επιλογής), μάθημα2 (υποχρεωτικό), μάθημα3."
---

**Ιστορικό Συνομιλίας (Μόνο για κατανόηση, όχι για απαντήσεις):**  
{history}

**Νέα Ερώτηση:**  
{question}

**Πληροφορίες από τη βάση γνώσεων:**  
{context}
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
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποιος είναι ο A;

                 - Προηγούμενη Απάντησή σου: Ο A είναι καθηγητής στο τμήμα Πληροφορικής. "  
- **Τελευταία ερώτηση:** "Πού βρίσκεται το γραφείο του;"  
- **Αναμενόμενη έξοδος:** `"Πού βρίσκεται το γραφείο του A;"`  

### Παράδειγμα 2  
- **Ιστορικό:** (Κενό)  
- **Τελευταία ερώτηση:** "Ποια μαθήματα διδάσκει ο B;"  
- **Αναμενόμενη έξοδος:** `"Ποια μαθήματα διδάσκει ο B;"`  

### Παράδειγμα 3  
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποιος είναι ο A;

                 - Προηγούμενη Απάντησή σου: Ο A είναι καθηγητής στο τμήμα Πληροφορικής. "  
- **Τελευταία ερώτηση:** "Ποια είναι τα μαθήματα του Τέταρτου Έτους;"  
- **Αναμενόμενη έξοδος:** `"Ποια είναι τα μαθήματα του Τέταρτου Έτους;"`

### Παράδειγμα 4
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Πόσες πιστωτικές μονάδες χρειάζονται για το πτυχίο;

                 - Προηγούμενη Απάντησή σου: Για το πτυχίο απαιτούνται 240 πιστωτικές μονάδες." 
- **Τελευταία ερώτηση:** "Πόσα μαθήματα πρέπει να επιλέξω στο 5ο εξάμηνο;"  
- **Αναμενόμενη έξοδος:** `"Πόσα μαθήματα πρέπει να επιλέξω στο 5ο εξάμηνο;"`  

### Παράδειγμα 5
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποια είναι τα υποχρεωτικά μαθήματα του 3ου εξαμήνου;

                 - Προηγούμενη Απάντησή σου: Τα υποχρεωτικά μαθήματα του 3ου εξαμήνου είναι το Α και το Β." 
- **Τελευταία ερώτηση:** "Υπάρχουν προαπαιτούμενα;"  
- **Αναμενόμενη έξοδος:** `"Υπάρχουν προαπαιτούμενα για τα υποχρεωτικά μαθήματα του 3ου εξαμήνου;"` 

### Παράδειγμα 6
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Τι είναι το Erasmus+ και πώς μπορώ να συμμετάσχω;

                 - Προηγούμενη Απάντησή σου: Το Erasmus+ είναι ένα πρόγραμμα ανταλλαγής φοιτητών." 
- **Τελευταία ερώτηση:** "Πόσο διαρκεί;"  
- **Αναμενόμενη έξοδος:** `"Πόσο διαρκεί το πρόγραμμα Erasmus+;"` 

### Παράδειγμα 7
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποια είναι η διαδικασία δήλωσης μαθημάτων;

                 - Προηγούμενη Απάντησή σου: Η διαδικασία δήλωσης μαθημάτων περιλαμβάνει την υποβολή αίτησης." 
- **Τελευταία ερώτηση:** "Μέχρι πότε μπορώ να δηλώσω;"  
- **Αναμενόμενη έξοδος:** `"Μέχρι πότε μπορώ να δηλώσω μαθήματα;"` 

### Παράδειγμα 8
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποιος είναι ο καθηγητής του μαθήματος Β;

                 - Προηγούμενη Απάντησή σου: Ο καθηγητής του μαθήματος Β είναι ο Γ." 
- **Τελευταία ερώτηση:** "Πού μπορώ να βρω το υλικό του;"  
- **Αναμενόμενη έξοδος:** `"Πού μπορώ να βρω το υλικό του μαθήματος Β;"` 

### Παράδειγμα 9
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Πώς μπορώ να πάρω απαλλαγή από ένα μάθημα;

                 - Προηγούμενη Απάντησή σου: Για απαλλαγή από ένα μάθημα, πρέπει να υποβάλεις αίτηση." 
- **Τελευταία ερώτηση:** "Τι ισχύει για τα εργαστήρια;"  
- **Αναμενόμενη έξοδος:** `"Τι ισχύει για τα εργαστήρια;"` 

### Παράδειγμα 10
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποια είναι τα υποχρεωτικά μαθήματα στο 3ο εξάμηνο;

                 - Προηγούμενη Απάντησή σου: Τα υποχρεωτικά μαθήματα στο 3ο εξάμηνο είναι το Α και το Β." 
- **Τελευταία ερώτηση:** "Και στο 5ο;"  
- **Αναμενόμενη έξοδος:** `"Ποια είναι τα υποχρεωτικά μαθήματα στο 5ο εξάμηνο;"` 

### Παράδειγμα 11
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Πόσες ώρες πρέπει να αφιερώνω για τα μαθήματα κάθε εβδομάδα;

                 - Προηγούμενη Απάντησή σου: Ο μέσος όρος είναι περίπου 25-30 ώρες την εβδομάδα για το πλήρες πρόγραμμα σπουδών." 
- **Τελευταία ερώτηση:** "Πόσες ώρες είναι για το 3ο εξάμηνο;"  
- **Αναμενόμενη έξοδος:** `"Πόσες ώρες πρέπει να αφιερώνω για το 3ο εξάμηνο;"` 

### Παράδειγμα 12
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποιος είναι ο επικεφαλής του τμήματος Πληροφορικής;

                 - Προηγούμενη Απάντησή σου: Ο επικεφαλής του τμήματος είναι ο Α" 
- **Τελευταία ερώτηση:** "Ποια είναι τα μαθήματα του τμήματος;"  
- **Αναμενόμενη έξοδος:** `"Ποια είναι τα μαθήματα του τμήματος Πληροφορικής;"` 

### Παράδειγμα 13
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποια είναι τα μαθήματα του 1ου εξαμήνου

                 - Προηγούμενη Απάντησή σου: Τα μαθήματα του 1ου εξαμήνου είναι 1. Α, 2. Β, 3. Γ, 4. Δ. , 5. Ε." 
- **Τελευταία ερώτηση:** "Πες μου πληροφορίες για το 3"  
- **Αναμενόμενη έξοδος:** `"Πες μου πληροφορίες για το μάθημα Γ."` 

### Παράδειγμα 14
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποια μαθήματα διδάσκει ο κ. Κουσιουρής;

                 - Προηγούμενη Απάντησή σου: Τα μαθήματα του καθηγητή κ. Κουσιουρής είναι 1. Α, 2. Β." 
- **Τελευταία ερώτηση:** "Ποια μαθήματα έχει ο κ. Ριζομυλιώτης;"  
- **Αναμενόμενη έξοδος:** `"Ποια μαθήματα διδάσκει ο Β;"` 

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