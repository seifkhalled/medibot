from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os, json, traceback
from typing import List, Tuple
from pydantic import BaseModel, Field

# LangChain imports
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever, Document

# Community utilities
from langchain_community.utilities import PubMedAPIWrapper

# Local imports
from src.prompt import system_prompt, history_instructions

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in environment variables.")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables.")

os.environ.setdefault("USER_AGENT", "medical-chatbot/1.0 (contact: youremail@example.com)")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# -----------------------------
# Red flags for urgent symptoms
# -----------------------------
RED_FLAGS = [
    "trouble breathing",
    "chest pain",
    "severe headache",
    "unconscious",
    "seizure",
    "vision loss",
    "severe vomiting"
]

def check_red_flags(user_text: str) -> bool:
    text = user_text.lower()
    return any(flag in text for flag in RED_FLAGS)

# -----------------------------
# Symptom progression tracker
# -----------------------------
class SymptomTracker:
    def __init__(self):
        self.state = {
            "main_symptom": None,
            "duration": None,
            "progression": None,
            "associated_symptoms": []
        }

    def update_state(self, user_message: str):
        pass

    def to_summary(self) -> str:
        parts = []
        if self.state["main_symptom"]:
            parts.append(f"Main symptom: {self.state['main_symptom']}")
        if self.state["duration"]:
            parts.append(f"Duration: {self.state['duration']}")
        if self.state["progression"]:
            parts.append(f"Progression: {self.state['progression']}")
        if self.state["associated_symptoms"]:
            parts.append(f"Associated symptoms: {', '.join(self.state['associated_symptoms'])}")
        return "; ".join(parts)

symptom_tracker = SymptomTracker()

# -----------------------------
# Embeddings & Vector store
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = "medicalbot-minilm"
docsearch = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
book_retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -----------------------------
# PubMed (real-time)
# -----------------------------
try:
    pubmed = PubMedAPIWrapper()
except Exception as e:
    print(f"[WARN] PubMed disabled: {e}")
    pubmed = None

# -----------------------------
# Hybrid retriever
# -----------------------------
class HybridRetriever(BaseRetriever):
    retriever_a: BaseRetriever
    retriever_b: object = None

    def _get_relevant_documents(self, query, run_manager=None):
        docs = []
        try:
            docs += self.retriever_a.get_relevant_documents(query)
        except Exception as e:
            print(f"[HybridRetriever] Pinecone error: {e}")

        if self.retriever_b:
            try:
                results = self.retriever_b.run(query)
                for r in results.split("\n"):
                    r = r.strip()
                    if r:
                        docs.append(Document(page_content=r, metadata={"source": "PubMed", "original_query": query}))
            except Exception as e:
                print(f"[HybridRetriever] PubMed error: {e}")
        return docs

    async def _aget_relevant_documents(self, query, run_manager=None):
        return self._get_relevant_documents(query, run_manager)

hybrid_retriever = HybridRetriever(retriever_a=book_retriever, retriever_b=pubmed)

# -----------------------------
# LLM
# -----------------------------
friendly_system_prompt = """
You are a friendly, empathetic, and knowledgeable AI assistant.
- Your primary role is to provide health information.
- Greet users warmly and remember conversational context like their name or previous questions.
- If the user asks a non-medical question, answer it concisely and politely, then gently guide the conversation back to their health concern.
- When providing medical information, maintain an empathetic, reassuring tone without losing accuracy.
- Avoid unnecessary jargon and explain concepts in plain language.
- Keep responses focused, clear, and easy to read.
""" + system_prompt

llm = ChatGroq(
    model_name="llama3-70b-8192",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3,
    max_tokens=700,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def format_chat_history_for_prompt(messages):
    lines = []
    for m in messages:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines[-12:])

# -----------------------------
# Pydantic model for structured output
# -----------------------------
class MedicalCase(BaseModel):
    is_medical_question: bool = Field(..., description="True if the user's message is a health or medical question, False otherwise.")
    age: int | None = Field(None, description="The person's age in years. Null if not provided.")
    sex: str | None = Field(None, description="The person's sex ('male' or 'female'). Null if not provided.")
    symptoms: List[str] = Field([], description="A list of specific symptoms mentioned by the user.")
    duration: str | None = Field(None, description="How long the symptoms have been present.")
    progression: str | None = Field(None, description="How the symptoms are progressing.")
    free_text_summary: str = Field(..., description="A short, free-text summary of the user's inquiry.")
    
def extract_case_json(user_text: str) -> dict:
    try:
        extraction_llm = llm.with_structured_output(MedicalCase)
        case = extraction_llm.invoke(f"Extract information from this message: {user_text}")
        return case.model_dump()
    except Exception as e:
        return {
            "is_medical_question": "symptom" in user_text.lower() or "health" in user_text.lower() or "medical" in user_text.lower(),
            "age": None,
            "sex": None,
            "symptoms": [],
            "duration": None,
            "progression": None,
            "free_text_summary": user_text[:500]
        }

# -----------------------------
# Build enumerated context with limited PubMed links
# -----------------------------
def build_enumerated_context(docs, case, k=3) -> Tuple[str, List[str]]:
    out = []
    links = []

    query_terms = []
    if case.get("symptoms"):
        query_terms.extend(case["symptoms"])
    if case.get("duration"):
        query_terms.append(case["duration"])
    pubmed_query = "+".join(query_terms) if query_terms else None
    pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/?term={pubmed_query}" if pubmed_query else None

    for i, d in enumerate(docs[:k], start=1):
        src = d.metadata.get("source") if isinstance(d.metadata, dict) else None
        tag = f"Source {i}" + (f" ({src})" if src else "")
        content = d.page_content[:1200]
        out.append(f"[{tag}] {content}")

    if pubmed_link:
        links.append(pubmed_link)

    return "\n\n".join(out), links

# -----------------------------
# Manual doctor recommendation
# -----------------------------
def recommend_doctor_manual(symptoms: List[str]) -> str:
    mapping = {
        "fever": "general physician",
        "cough": "pulmonologist",
        "headache": "neurologist",
        "rash": "dermatologist",
        "stomach pain": "gastroenterologist",
        "joint pain": "rheumatologist"
    }
    recommended = set()
    for s in symptoms:
        for key, doc in mapping.items():
            if key in s.lower():
                recommended.add(doc)
    if recommended:
        return "Based on your symptoms, you may consider consulting: " + ", ".join(recommended)
    return ""

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        # -----------------------------
        # Red flag detection
        # -----------------------------
        if check_red_flags(message):
            answer = (
                "⚠️ This is a serious symptom. You should seek **emergency medical attention immediately**. "
                "Call your local emergency number (e.g., 911 in the U.S.) or go to the nearest ER.\n\n"
                "**Disclaimer:** I'm an AI assistant and not a licensed clinician. "
                "This advice does not replace professional medical care."
            )
            return jsonify({"response": answer})

        # -----------------------------
        # Extract structured medical info
        # -----------------------------
        case = extract_case_json(message)
        if not case.get("is_medical_question"):
            history = memory.load_memory_variables({}).get("chat_history", [])
            history_text = format_chat_history_for_prompt(history)
            
            non_medical_prompt = ChatPromptTemplate.from_messages([
                ("system", friendly_system_prompt),
                ("human",
                 "User question:\n{question}\n\nChat history:\n{chat_history}\n\n"
                 "Respond concisely to this non-medical question, then guide back to a medical topic."
                )
            ])
            msgs = non_medical_prompt.format_messages(question=message, chat_history=history_text)
            response = llm.invoke(msgs)
            answer = response.content.strip()
            memory.save_context({"input": message}, {"output": answer})
            return jsonify({"response": answer})

        # -----------------------------
        # Update symptom tracker
        # -----------------------------
        symptom_tracker.state["main_symptom"] = case.get("symptoms")[0] if case.get("symptoms") else None
        symptom_tracker.state["duration"] = case.get("duration")
        symptom_tracker.state["progression"] = case.get("progression")
        symptom_tracker.state["associated_symptoms"].extend(s for s in case.get("symptoms", []) if s != symptom_tracker.state["main_symptom"])
        
        case_summary = symptom_tracker.to_summary()

        keywords = " ".join(case.get("symptoms", []) or [])
        duration = case.get("duration") or ""
        retrieval_query = f"{message}\nSymptoms: {keywords}\nDuration: {duration}".strip()

        # -----------------------------
        # Retrieve context and links
        # -----------------------------
        docs = hybrid_retriever.get_relevant_documents(retrieval_query)
        context_text, reference_links = build_enumerated_context(docs, case, k=3)

        history = memory.load_memory_variables({}).get("chat_history", [])
        history_text = format_chat_history_for_prompt(history)

        final_prompt_template = ChatPromptTemplate.from_messages([
            ("system", friendly_system_prompt + "\n" + history_instructions),
            ("human",
             "Start by acknowledging the user in a friendly way before giving medical reasoning.\n"
             "User question:\n{question}\n\nStructured case JSON:\n{case_json}\n\n"
             "Case progression summary:\n{case_summary}\n\n"
             "Retrieved context (enumerated):\n{context}\n\n"
             "Write the answer with:\n"
             "- A short, empathetic case summary in plain language\n"
             "- 2–4 possible categories (not diagnoses) with rationale\n"
             "- Simple next steps\n"
             "- Red flags\n"
             "- Map statements to sources\n"
             "- Include references/links if available\n"
             "- End with disclaimer"
            ),
        ])

        msgs = final_prompt_template.format_messages(
            question=message,
            case_json=json.dumps(case, ensure_ascii=False),
            case_summary=case_summary,
            context=context_text,
            chat_history=history_text
        )

        response = llm.invoke(msgs)
        answer = response.content.strip()

        doctor_rec = recommend_doctor_manual(case.get("symptoms", []))
        if doctor_rec:
            answer += "\n\n" + doctor_rec
        
        if reference_links:
            answer += "\n\n**References:**\n"
            for link in reference_links:
                answer += f"- {link}\n"
        
        memory.save_context({"input": message}, {"output": answer})

        return jsonify({"response": answer})

    except Exception as e:
        traceback.print_exc()
        if "503" in str(e) or "Service unavailable" in str(e):
            return jsonify({"response": "The model is temporarily unavailable. Please try again in a moment."})
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
