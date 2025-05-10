import time
import os
import json
import sys
from fastapi import FastAPI, Query, BackgroundTasks
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
from langchain.prompts import PromptTemplate
import torch
import gc
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
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
    
class MarkdownTitleTextSplitterMainHua(TextSplitter):
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
            elif line.startswith("=="):

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
                result = f"Οι {node} του τμήματος Πληροφορικής και τηλεματικής του Προπτυχιακού Πρόγραμμα Σπουδών είναι: -"+'\n-  '.join(neighbors) + "\n"
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

def rerank(query, documents):
    pairs = [(query, doc) for doc in documents]

    try:
        ranked_indexes = []
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)

        torch.cuda.empty_cache()

        with torch.no_grad():
            scores = reranker_model(**inputs).logits.view(-1).cpu().tolist()  #Get scores

        #Sort documents by score and store indexes
        ranked_indexes = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    except Exception as e:
        print(f"Error during reranking: {e}")
        return [0,1,2,3,4]
    return ranked_indexes


def rephrase_query(query, history):
    print(f"Original Query: {query}\n\n")
    rephrase_prompt_format = rephrase_prompt.format(history=history, query=query)
    rephraseResponse = chatModel.create_chat_completion(messages=[{"role": "system", "content": rephrase_prompt_format}])

    match = re.search(r'"([^"]*)"', rephraseResponse['choices'][0]['message']['content'])
    if match:
        query = match.group(1)
    print(f"Rephrased Query: {query}\n\n")
    return query


def split_query(query):
    split_prompt_format = split_prompt.format(question=query)
    splitResponse = chatModel.create_chat_completion(messages=[{"role": "system", "content": split_prompt_format}])
    print(splitResponse['choices'][0]['message']['content'])
    questions = re.findall(r'<(.*?)>', (splitResponse['choices'][0]['message']['content']))
    if len(questions) == 0 or not questions:
        return [query]
    else:
        return questions

def reRankingRetriever_local(query, retriever):
    retrieved_documents = retriever.invoke(query)
    documents = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in retrieved_documents]

    ranked_indexes = rerank(query, documents)
    
    ranked_indexes = ranked_indexes[:5]
    filtered_documents = [retrieved_documents[i] for i in ranked_indexes]

    return filtered_documents



def query_model(question, docs, chatModel, history, show_prompt=True, docs_need_format = True):
    if docs_need_format:
        formatted_docs = "\n".join([f"\n\t{i+1} - {doc.page_content}" for i, doc in enumerate(docs)])
    else:
        formatted_docs = docs
    prompt_text = rag_prompt.format(question=question, context=formatted_docs, history=history) 
    if show_prompt:
        print(prompt_text)
    response = chatModel.create_chat_completion(messages=[{"role": "user", "content": prompt_text}])
    if show_prompt:
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
        return ""

    # Traverse from end to start, looking for the last valid user→bot pair
    for i in range(len(chat_history) - 2, -1, -1):
        if (chat_history[i]["role"] == "user" and
            chat_history[i + 1]["role"] == "bot"):
            
            last_user_question = chat_history[i]["content"]
            last_bot_answer = chat_history[i + 1]["content"]

            return (f"- Προηγούμενη Ερώτηση από χρήστη: {last_user_question}\n\n"
                    f"- Προηγούμενη Απάντησή σου: {last_bot_answer}")

    return ""



from typing import List, Dict
import asyncio
queue = asyncio.Queue()
processing_sessions = set()
session_last_seen = {}

INACTIVITY_TIMEOUT = 5
async def remove_stale_sessions():
    while True:
        now = time.time()
        stale_sessions = [
            sid for sid, last_seen in session_last_seen.items()
            if now - last_seen > INACTIVITY_TIMEOUT and sid not in processing_sessions
        ]

        for sid in stale_sessions:
            # Remove from queue if exists
            try:
                queue._queue.remove(sid)
                print(f"Session {sid} removed from queue due to inactivity.")
            except ValueError:
                pass
            processing_sessions.discard(sid)
            session_last_seen.pop(sid, None)

        await asyncio.sleep(5)  #Run every 5 seconds


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(remove_stale_sessions())

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]]
    session_id: str


@app.post("/chat")
async def run_chat(request: ChatRequest):
    message, history, session_id = request.message, request.history, request.session_id
    session_last_seen[session_id] = time.time()
    print(f"Queue content: {queue._queue}")
    print(f"Processing sessions: {processing_sessions}")

    # Return position in queue if already in progress or queued
    if session_id in processing_sessions:
        position = list(queue._queue).index(session_id) + 1 if session_id in queue._queue else None
        return {"message": f"⏳ Παρακαλώ περιμένετε... Είστε στη θέση {position} της ουράς."}
    if session_id not in queue._queue:
        await queue.put(session_id)
    position = list(queue._queue).index(session_id)

    # If not first in queue, return queue position
    if position > 0:
        if position == 1:
            return {"message": f"⏳ Παρακαλώ περιμένετε... Είστε στη θέση {position} της ουράς και έχετε προτεραιότητα!"}
        else:
            return {"message": f"⏳ Παρακαλώ περιμένετε... Είστε στη θέση {position} της ουράς."}
        

    processing_sessions.add(session_id)

    try:
        def process_request():
            start = time.perf_counter()

            formatted_history = format_history(history)
            if formatted_history == "":
                formatted_history = history
            new_question = rephrase_query(message, formatted_history)
            questions = split_query(new_question)[:5]

            if len(questions) == 1:
                documents = reRankingRetriever_local(new_question, retriever)
                answer = query_model(new_question, documents, chatModel, formatted_history)
            else:
                answers_list = []
                for question in questions:
                    documents = reRankingRetriever_local(question, retriever)
                    partial_answer = query_model(question, documents, chatModel, formatted_history, show_prompt=False)
                    answers_list.append(partial_answer)
                context = "\n".join(answers_list)
                answer = query_model(
                    question=new_question,
                    chatModel=chatModel,
                    docs=context,
                    history=formatted_history,
                    show_prompt=True,
                    docs_need_format=False
                )

            torch.cuda.empty_cache()
            gc.collect()
            end = time.perf_counter()
            print(f"Time to run query: {end - start}")
            return answer

        # Run the blocking task in a background thread
        generated_answer = await asyncio.to_thread(process_request)
        processing_sessions.remove(session_id)
        queue._queue.remove(session_id)
        return {"message": generated_answer}

    except Exception as e:
        processing_sessions.remove(session_id)
        queue._queue.remove(session_id)
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
    directory_path_main_hua = '/home/it2021087/chatBot/Hua/scraped_pages_el_main_Hua'

    loader = DirectoryLoader(directory_path, glob="*.md", loader_cls=TextLoader)
    documents = loader.load()

    loader_main_hua = DirectoryLoader(directory_path_main_hua, glob="*.md", loader_cls=TextLoader)
    documents_main_hua = loader_main_hua.load()
    print(f"Loaded {len(documents)} documents")

    #Apply the custom Markdown splitter
    markdown_splitter = MarkdownTitleTextSplitter()
    markdown_splitter_main_hua = MarkdownTitleTextSplitterMainHua()
    texts = markdown_splitter.split_documents(documents)
    texts_main_hua = markdown_splitter_main_hua.split_documents(documents_main_hua)

    #Combine the two lists of documents
    texts.extend(texts_main_hua)

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
- **Ιστορικό:** "- Προηγούμενη Ερώτηση από χρήστη: Ποια είναι τα μαθήματα του πρώτου εξαμήνου;

                 - Προηγούμενη Απάντησή σου: Τα μαθήματα του πρώτου εξαμήνου  είναι: Υπολογιστικά Μαθηματικά, Ψηφιακή Τεχνολογία και Εφαρμογές Τηλεματικής, Προγραμματισμός Ι, Λογική Σχεδίαση, Διακριτά Μαθηματικά." 
- **Τελευταία ερώτηση:** "Ποιος διδάσκει το καθένα από αυτά;"  
- **Αναμενόμενη έξοδος:** `"Ποιος διδάσκει τα μαθήματα Υπολογιστικά Μαθηματικά, Ψηφιακή Τεχνολογία και Εφαρμογές Τηλεματικής, Προγραμματισμός Ι, Λογική Σχεδίαση, Διακριτά Μαθηματικά;"` 

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

### Παράδειγμα 15
- **Ιστορικό:** (κενό)
- **Τελευταία ερώτηση:** "Ποια είναι τα μαθήματα του πρώτου έτους;"  
- **Αναμενόμενη έξοδος:** `"Ποια είναι τα μαθήματα του πρώτου έτους;"`

Ιστορικό Συνομιλίας: {history}  
Τελευταία ερώτηση: {query}  

Επαναδιατυπωμένο Κείμενο:  
"""

    rephrase_prompt = PromptTemplate(
        input_variables=["history", "query"],
        template=template_rephrase
    )





    split_template = """Είσαι ένας αυστηρός επεξεργαστής ερωτήσεων που λαμβάνει μία σύνθετη ερώτηση και την αναλύει σε ανεξάρτητα, αυτόνομα ερωτήματα. Ο στόχος σου είναι να διασφαλίσεις ότι κάθε ερώτηση αφορά **ένα μόνο αντικείμενο** και δεν περιέχει **πολλαπλά ζητούμενα** στην ίδια πρόταση.

## **Κανόνες Διάσπασης Ερωτήσεων**:
1. **Απαγορεύεται** να επιστρέψεις μία ερώτηση που περιλαμβάνει περισσότερα από ένα ξεχωριστά αντικείμενα ή έννοιες.
2. Αν η αρχική ερώτηση περιέχει περισσότερα από ένα ζητούμενα (π.χ., "Ποιος διδάσκει Τεχνητή Νοημοσύνη και Διακριτά Μαθηματικά;"), τότε **πρέπει** να τη χωρίσεις σε **δύο ή περισσότερες** ερωτήσεις, καθεμία με **ένα μόνο ζητούμενο**.
3. Οι νέες ερωτήσεις πρέπει να είναι **αυτοτελείς** και να **διατηρούν το νόημα** της αρχικής ερώτησης.
4. Δεν επιτρέπεται καμία αυθαίρετη αλλαγή ή αφαίρεση πληροφορίας από την αρχική ερώτηση.
5. Πρέπει να τηρείται **σαφής και φυσική διατύπωση** στις νέες ερωτήσεις, χωρίς ασαφείς αναφορές (π.χ., **όχι** "Και το άλλο;").
6. Αν η ερώτηση είναι ήδη **απλή και αυτοτελής**, την επιστρέφεις **ως έχει** χωρίς τροποποίηση.
8. Ο όρος **"Πληροφορική και Τηλεματική" είναι ένας όρος** και δεν πρέπει να διασπαστεί.
7. **Κάθε νέα ερώτηση ΠΡΕΠΕΙ να είναι σε αυστηρή μορφή: <Νέα Ερώτηση 1> <Νέα Ερώτηση 2> ...**  
   - **Απαγορεύεται οποιαδήποτε άλλη μορφή.**
   - **Οι απαντήσεις πρέπει να είναι αποκλειστικά μέσα σε γωνιακές αγκύλες (`< >`).**
   - **Καμία πρόσθετη πληροφορία ή εξήγηση δεν επιτρέπεται.** 
   - **Οποιαδήποτε άλλη μορφή θα θεωρείται ΛΑΘΟΣ** 

---

## **Παραδείγματα Διάσπασης**:

### Παράδειγμα 1:
**Είσοδος:**  
*"Ποιος διδάσκει Τεχνητή Νοημοσύνη και Διακριτά Μαθηματικά;"*

**Έξοδος:**  
`<Ποιος διδάσκει Τεχνητή Νοημοσύνη;> <Ποιος διδάσκει Διακριτά Μαθηματικά;>`

---

### Παράδειγμα 2:
**Είσοδος:**  
*"Ποια είναι τα υποχρεωτικά μαθήματα του 3ου και του 5ου εξαμήνου;"*

**Έξοδος:**  
`<Ποια είναι τα υποχρεωτικά μαθήματα του 3ου εξαμήνου;> <Ποια είναι τα υποχρεωτικά μαθήματα του 5ου εξαμήνου;>`

---

### Παράδειγμα 3:
**Είσοδος:**  
*"Ποιος είναι ο καθηγητής του μαθήματος Β και πού μπορώ να βρω το υλικό του;"*

**Έξοδος:**  
`<Ποιος είναι ο καθηγητής του μαθήματος Β;> <Πού μπορώ να βρω το υλικό του μαθήματος Β;>`

---

### Παράδειγμα 4:
**Είσοδος:**  
*"Ποιες είναι οι προϋποθέσεις για το μάθημα Α και το μάθημα Β;"*

**Έξοδος:**  
`<Ποιες είναι οι προϋποθέσεις για το μάθημα Α;> <Ποιες είναι οι προϋποθέσεις για το μάθημα Β;>`

---

### Παράδειγμα 5:
**Είσοδος:**  
*"Ποιο είναι το πρόγραμμα μαθημάτων του 3ου και του 4ου εξαμήνου;"*

**Έξοδος:**  
`<Ποιο είναι το πρόγραμμα μαθημάτων του 3ου εξαμήνου;> <Ποιο είναι το πρόγραμμα μαθημάτων του 4ου εξαμήνου;>`

---

### Παράδειγμα 6:
**Είσοδος:**  
*"Πόσες πιστωτικές μονάδες απαιτούνται για το πτυχίο;"*

**Έξοδος:**  
`<Πόσες πιστωτικές μονάδες απαιτούνται για το πτυχίο;>`  

(Δεν χωρίζεται γιατί έχει **ένα** μόνο ζητούμενο.)

---

### Παράδειγμα 7:
**Είσοδος:**  
*"Ποιος διδάσκει τα μαθήματα Προγραμματισμός Ι, Διακριτά Μαθηματικά, Λογική Σχεδίαση, Ψηφιακή Τεχνολογία και Εφαρμογές Τηλεματικής, Υπολογιστικά Μαθηματικά;"*

**Έξοδος:**  
`<Ποιος διδάσκει το μάθημα Προγραμματισμός Ι> <Ποιος διδάσκει το μάθημα Διακριτά Μαθηματικά;> <Ποιος διδάσκει το μάθημα Λογική Σχεδίαση> <Ποιος διδάσκει το μάθημα Ψηφιακή Τεχνολογία και Εφαρμογές Τηλεματικής> <Ποιος διδάσκει το μάθημα Υπολογιστικά Μαθηματικά;>`

---

### Παράδειγμα 8:
**Είσοδος:**  
*"Ποιες είναι οι προϋποθέσεις για την απόκτηση πτυχίου;"*

**Έξοδος:**  
`<Ποιες είναι οι προϋποθέσεις για την απόκτηση πτυχίου;>`  

(Δεν χωρίζεται γιατί έχει **ένα** μόνο ζητούμενο.)

---

### Παράδειγμα 9:
**Είσοδος:**  
*"Ποια είναι τα μαθήματα του πρώτου έτους;"*

**Έξοδος:**  
`<Ποια είναι τα μαθήματα του πρώτου έτους;>`  

(Δεν χωρίζεται γιατί έχει **ένα** μόνο ζητούμενο.)

---

### Παράδειγμα 9:
**Είσοδος:**  
*"Ποιος είναι ο κοσμήτορας του τμήματος Πληροφορικής και Τηλεματικής;"*

**Έξοδος:**  
`<Ποιος είναι ο κοσμήτορας του τμήματος Πληροφορικής και Τηλεματικής;>`  

(Δεν χωρίζεται γιατί η πληροφορική και τηλεματική είναι **ένας** όρος.)

---

### Παράδειγμα 10:
**Είσοδος:**  
*"Που βρισκεται το τμημα πληροφορικης και τηλεματικης "*

**Έξοδος:**  
`<που βρισκεται το τμημα πληροφορικης και τηλεματικης >`  

(Δεν χωρίζεται γιατί η πληροφορική και τηλεματική είναι **ένας** όρος.)

---

**Αρχική Ερώτηση:**  
{question}  

**Αναλυμένες Ερωτήσεις:**  

"""

    split_prompt = PromptTemplate(
        input_variables=["question"],
        template=split_template
    )

    
    print(f"Loaded chat model")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)