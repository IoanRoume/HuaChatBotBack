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


class MarkdownTitleTextSplitter(TextSplitter):
    def split_text(self, text: str):
        sections = []
        current_section = []
        previous_line = ""
        skipLine = False

        for line in text.splitlines():
            # Check if the line starts with a Markdown header (e.g., #, ##, ###)
            if line == "":
              continue
            elif line.startswith("#"):
                current_section.append(previous_line)
                if current_section:  # Save the current section if it exists
                    sections.append("\n".join(current_section))
                current_section = [line]  # Start a new section
                skipLine = True
            # Check if the line starts with '**' (indicating bold text or a special marker)
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
            # Check if the line is a separator (a line with only '=' characters)
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
        # Add the final section if there's remaining content
        if current_section:
            sections.append("\n".join(current_section))

        return sections


def rerank(query, documents):
    pairs = [(query, doc) for doc in documents]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.view(-1).cpu().tolist()  # Get scores

    # Sort documents by score (higher is better) and store indexes
    ranked_indexes = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return ranked_indexes


def reRankingRetriever_local(query, retriever):
    retrieved_documents = retriever.invoke(query)
    documents = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in retrieved_documents]

    ranked_indexes = rerank(query, documents)

    ranked_indexes = ranked_indexes[:5]

    filtered_documents = [retrieved_documents[i] for i in ranked_indexes]
    return filtered_documents


def query_model(question, docs, chatModel):

    prompt_text = rag_prompt.format(question=question, context=docs)

    response = chatModel.create_chat_completion(messages=[{"role": "user", "content": prompt_text}])



    return response['choices'][0]['message']['content']





app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_request_time = time.time()
inactive_time = 3600
allowCleanup = True
areModelsWiped = False
lock = threading.Lock()

def check_inactivity():
    """Background thread to check for inactivity."""
    global last_request_time
    while True:
        with lock:
            elapsed_time = time.time() - last_request_time
            if elapsed_time > inactive_time and allowCleanup:
                print("No activity for 1 Hour. Cleaning Cache...", file=sys.stderr)
                cleanup()
        time.sleep(900)


def cleanup():
    global areModelsWiped, tokenizer, reranker_model, retriever, vectordb, embedding, allowCleanup, chatModel
    torch.cuda.empty_cache()
    gc.collect()
    allowCleanup = False
    del tokenizer, reranker_model, retriever, vectordb, embedding, chatModel
    areModelsWiped = True

@app.middleware("http")
async def update_last_request_time(request: Request, call_next):
    global last_request_time, allowCleanup, areModelsWiped
    allowCleanup = True
    last_request_time = time.time()
    return await call_next(request)


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


def load_models():
    global areModelsWiped, tokenizer,reranker_model, retriever, vectordb, embedding, chatModel
    with lock:
        if not areModelsWiped:
            return
        
        print(f"Reinitializing models...", file=sys.stderr)

        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True)       
        reranker_model = AutoModelForSequenceClassification.from_pretrained("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")

        persist_directory = 'db_el'
        embedding = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v2-moe",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': False}
        )

        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        retriever = vectordb.as_retriever(search_kwargs={"k": 40})

        chatModel = Llama.from_pretrained(repo_id="bartowski/ilsp_Llama-Krikri-8B-Instruct-GGUF", filename="ilsp_Llama-Krikri-8B-Instruct-Q4_K_M.gguf", n_ctx=8192, n_gpu_layers=-1)

        areModelsWiped = False
        print("Models reinitialized", file=sys.stderr)


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def run_chat(request: ChatRequest):
    #global chatModel
    load_models()
    try:
        #chatModel = Llama.from_pretrained(repo_id=repo_id_chat_model, filename=filename_chat_model, n_ctx=8192, n_gpu_layers=-1)
        documents = reRankingRetriever_local(request.message, retriever)
        generated_answer = query_model(request.message, documents, chatModel)
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
                    f.write(f"Time: ** {datetime.now()}\n\n")
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
            f.write(f"Time: ** {datetime.now()}\n\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")
    return {"message": "Feedback saved!", "id": user_id}


thread = threading.Thread(target=check_inactivity, daemon=True)
thread.start()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    directory_path = '/home/it2021087/chatBot/scraped_pages_el'

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

    Ερώτηση: {question}

    Πληροφορίες από τη βάση γνώσεων:
    {context}

    Απάντηση:
    """

    rag_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=template
    )
    print(f"Loaded chat model")


    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)