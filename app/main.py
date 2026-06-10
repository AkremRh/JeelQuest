import os
import io
import re
import shutil
import unicodedata
import smtplib
from datetime import datetime, timedelta, timezone
from typing import List
from contextlib import asynccontextmanager
import asyncio

# --- FRAMEWORK WEB & API ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- TRAITEMENT DE DONNÉES & VISUALISATION ---
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force un backend non-interactif pour éviter les crashs en environnement conteneurisé (Docker)
import matplotlib.pyplot as plt
import seaborn as sns
from bson import ObjectId

# --- GÉNÉRATION DE RAPPORTS & MESSAGERIE ---
from fpdf import FPDF
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from apscheduler.schedulers.background import BackgroundScheduler

# --- BASE DE DONNÉES & ECOSYSTÈME IA ---
from pymongo import MongoClient
from pymilvus import connections, Collection, utility
import fitz  # PyMuPDF
from dotenv import load_dotenv

# --- INTÉGRATION LANGCHAIN & GEMINI ---
from google.genai import Client  # SDK Google GenAI natif pour l'agent analytique
from langchain_community.vectorstores import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

load_dotenv()

# --- CHARGEMENT DES COMPOSANTS ET CONFIGURATIONS ---
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_URI_1 = os.getenv("MONGO_URI_1")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")      
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2")    # Clé dédiée aux embeddings LangChain


TARGET_UNIVERSITY_ID = "6990b6d7c0e708ede1ed0178"

# Initialisation du client pour l'agent analytique (Rapports)
client_ai = Client(api_key=GOOGLE_API_KEY)

# Initialisation du modèle d'embeddings (Questy RAG)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=GOOGLE_API_KEY2
)

vectorstore = None

# --- INITIALISATION ET SÉCURISATION DU VECTORSTORE ---
def setup_vectorstore():
    """Initialise, vérifie la cohérence dimensionnelle et charge la collection Milvus/Zilliz"""
    global vectorstore
    try:
        connections.connect(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        test_vector = embedding_model.embed_query("test")
        expected_dim = len(test_vector)
        print(f"[+] Dimension attendue pour les embeddings : {expected_dim}")
        
        if utility.has_collection(COLLECTION_NAME):
            collection = Collection(COLLECTION_NAME)
            collection.load()
            schema = collection.schema
            vector_field = None
            for field in schema.fields:
                if field.name == "vector" or field.dtype == 101:  # 101 = FloatVector
                    vector_field = field
                    break
            
            if vector_field:
                current_dim = vector_field.params.get('dim', 0)
                print(f"[+] Dimension actuelle détectée dans Milvus : {current_dim}")
                if current_dim != expected_dim:
                    print(f" Incohérence de dimension ({current_dim} vs {expected_dim}). Réalignement de la collection...")
                    collection.drop()
                    utility.drop_collection(COLLECTION_NAME)
            else:
                print("[-] Champ vectoriel manquant dans le schéma actuel.")

        vectorstore = Milvus(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN},
            auto_id=True
        )
        print(f"[+] Vectorstore configuré avec succès sur la collection : {COLLECTION_NAME}")
        return vectorstore
    except Exception as e:
        print(f"[-] Erreur setup_vectorstore (Exécution du fallback direct) : {e}")
        vectorstore = Milvus(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": ZILLIZ_URI, "token": ZILLIZ_TOKEN},
            auto_id=True
        )
        return vectorstore


# --- LOGIQUE ET FORMATAGE DU PIPELINE DE COMPILATION DE RAPPORT (TALENTYZ) ---
def clean_pdf_text(text):
    if not text: return ""
    text = str(text)
    return re.sub(r'[^\x20-\x7EàâäéèêëîïôöùûüçÇÉÈÀÙ]', '', text).strip()

class CorporatePDF(FPDF):
    def header(self):
        self.set_fill_color(31, 41, 55)
        self.rect(0, 0, 210, 35, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.set_y(10)
        self.cell(0, 10, "TALENTYZ - BUSINESS ANALYTICS & ENGAGEMENT REPORT", align="L")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 10, f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", align="R")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(156, 163, 175)
        self.cell(0, 10, f"TALENTYZ Agent Report - Page {self.page_no()}/{{nb}} - Confidential", align="C")

def generate_ai_academic_summary(users_df, quests_perf_str, total_quests_count):
    total_students = len(users_df)
    avg_xp = users_df['xp'].mean() if 'xp' in users_df.columns else 0

    prompt = f"""
    You are an expert academic data analyst and student success consultant.
    Analyze the following consolidated matrix for University 'Talentyz'.
    
    KEY METRICS:
    - Total Monitored Students: {total_students}
    - Average Experience Points (XP): {avg_xp:.1f}
    - Total Available Academic Quests: {total_quests_count}

    TOP QUESTS PERFORMANCE (With Real Titles):
    {quests_perf_str}

    Generate a detailed executive analysis in English. 
    Structure it into two clean sections:
    1. GLOBAL ENGAGEMENT DIAGNOSTIC (Analyze student dynamics based on total numbers and events)
    2. QUEST PERFORMANCE AND ANALYSIS (Interpret which quests perform best based on their titles and engagement patterns)

    CRITICAL INSTRUCTION: Do NOT use any Markdown formatting (*, #, etc.). 
    Use Standard Sentence Case. Do NOT write headers or whole paragraphs in block CAPITAL LETTERS.
    """
    response = client_ai.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config={"temperature": 0.3, "max_output_tokens": 2000}
    )
    return response.text

def generate_report_and_send_email():
    """Génère les graphiques d'activité, sollicite l'analyse de Gemini et distribue le PDF par email"""
    try:
        print("[+] Initialisation du cycle analytique hebdomadaire...")
        client_mongo = MongoClient(MONGO_URI_1)
        db = client_mongo["pfe"]
        
        obj_university_id = ObjectId(TARGET_UNIVERSITY_ID) if ObjectId.is_valid(TARGET_UNIVERSITY_ID) else None
        query_user = {"$or": [{"universityId": TARGET_UNIVERSITY_ID}, {"universityId": obj_university_id}]}
        
        users_list = list(db["users"].find(query_user))
        if not users_list:
            print("[-] Extraction impossible : Aucun utilisateur référencé pour cette entité.")
            return
            
        df_users = pd.DataFrame(users_list)
        if 'xp' in df_users.columns:
            df_users['xp'] = pd.to_numeric(df_users['xp'], errors='coerce').fillna(0)
            df_users = df_users.sort_values(by='xp', ascending=False)
        
        user_name_map = {}
        for u in users_list:
            u_id_str = str(u['_id'])
            full_name = f"{u.get('firstName', '')} {u.get('lastName', '')}".strip()
            user_name_map[u_id_str] = full_name if full_name else "Unknown Student"

        student_ids = [uid for uid in df_users['_id'].tolist()] + [str(uid) for uid in df_users['_id'].tolist()]
        quests_list = list(db["userquests"].find({"student": {"$in": student_ids}}))
        df_quests = pd.DataFrame(quests_list) if quests_list else pd.DataFrame()
        
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        one_week_ago = now - timedelta(days=7)
        
        def extract_date_from_oid(oid):
            try:
                if isinstance(oid, ObjectId): return oid.generation_time.replace(tzinfo=None)
                elif ObjectId.is_valid(str(oid)): return ObjectId(str(oid)).generation_time.replace(tzinfo=None)
            except: pass
            return now

        achievement_winners = {}
        df_top_quests = pd.DataFrame()
        quests_summary_str = "No quest data available."
        newquests_list = []
        
        if not df_quests.empty:
            df_quests['creation_date'] = df_quests['_id'].apply(extract_date_from_oid)
            unique_quest_ids = df_quests['quest'].dropna().unique()
            mongo_quest_ids = [ObjectId(qid) for qid in unique_quest_ids if ObjectId.is_valid(qid)] + [str(qid) for qid in unique_quest_ids]
            
            newquests_list = list(db["newquests"].find({"_id": {"$in": mongo_quest_ids}}))
            
            if newquests_list:
                quest_mapping = {str(nq['_id']): nq.get('title', 'Untitled Quest') for nq in newquests_list}
                quest_achievements_map = {str(nq['_id']): nq.get('achievements', []) for nq in newquests_list}
                df_quests['quest_str'] = df_quests['quest'].astype(str)
                df_quests['quest_title'] = df_quests['quest_str'].map(quest_mapping).fillna(df_quests['quest_str'])
            else:
                df_quests['quest_title'] = df_quests['quest'].astype(str)
                quest_achievements_map = {}

            df_quests['completed'] = df_quests['completed'].astype(bool) if 'completed' in df_quests.columns else False
            
            df_weekly_quests = df_quests[df_quests['creation_date'] >= one_week_ago].copy()
            if not df_weekly_quests.empty:
                completed_weekly = df_weekly_quests[df_weekly_quests['completed'] == True]
                for _, row_q in completed_weekly.iterrows():
                    q_id = str(row_q['quest_str'])
                    student_id = str(row_q['student'])
                    student_name = user_name_map.get(student_id, "Unknown Student")
                    ach_list = quest_achievements_map.get(q_id, [])
                    for ach in ach_list:
                        if ach not in achievement_winners:
                            achievement_winners[ach] = set()
                        achievement_winners[ach].add(student_name)
                
            quest_group = df_quests.groupby('quest_title').agg(
                total_attempts=('student', 'count'),
                completed_count=('completed', 'sum'),
                unique_students=('student', 'nunique')
            ).reset_index()
            
            quest_group['completion_rate'] = (quest_group['completed_count'] / quest_group['total_attempts']) * 100
            quest_group['engagement_rate'] = (quest_group['unique_students'] / len(df_users)) * 100
            df_top_quests = quest_group.sort_values(by=['completion_rate', 'engagement_rate'], ascending=False).head(5)
            
            quests_summary_str = ""
            for _, r in df_top_quests.iterrows():
                quests_summary_str += f"- Quest Title: {r['quest_title']} | Completion Rate: {r['completion_rate']:.1f}% | Engagement Rate: {r['engagement_rate']:.1f}%\n"

        total_available_quests = len(newquests_list)

        summary_text = generate_ai_academic_summary(df_users, quests_summary_str, total_available_quests)
        summary_text_cleaned = clean_pdf_text(summary_text)

        # Génération des représentations visuelles
        sns.set_theme(style="whitegrid")
        df_top5_users = df_users.head(5).copy()
        df_top5_users['clean_name'] = df_top5_users.apply(lambda r: clean_pdf_text(f"{r.get('firstName', '')} {r.get('lastName', '')}"), axis=1)
        
        fig, ax1 = plt.subplots(figsize=(7.5, 3.8))
        bars_u = ax1.barh(df_top5_users['clean_name'], df_top5_users['xp'], color='#4f46e5', height=0.5)
        ax1.invert_yaxis()
        ax1.set_title("GENERAL INTERN EXPERIENCE LEADERBOARD (XP)", fontsize=12, fontweight='bold')
        for bar in bars_u:
            width = bar.get_width()
            ax1.text(width + 2, bar.get_y() + bar.get_height()/2, f'{int(width)} XP', va='center', fontweight='bold')
        plt.tight_layout()
        user_chart_path = "temp_user_xp.png"
        fig.savefig(user_chart_path, dpi=250)
        plt.close(fig)

        fig, ax2 = plt.subplots(figsize=(7.5, 3.8))
        if not df_top_quests.empty:
            df_top_quests['clean_title'] = df_top_quests['quest_title'].apply(lambda x: clean_pdf_text(x)[:26] + "...")
            ax2.bar(df_top_quests['clean_title'], df_top_quests['engagement_rate'], color='#0284c7', alpha=0.8, width=0.35, label="Engagement Rate")
            ax2.plot(df_top_quests['clean_title'], df_top_quests['completion_rate'], color='#10b981', marker='o', linewidth=2.5, label="Completion Rate")
        ax2.set_title("COMPARATIVE QUEST IMPACT AND COMPLETION METRICS", fontsize=12, fontweight='bold')
        ax2.legend()
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        quest_chart_path = "temp_quest_perf.png"
        fig.savefig(quest_chart_path, dpi=250)
        plt.close(fig)

        # Compilation FPDF
        pdf = CorporatePDF()
        pdf.alias_nb_pages()
        
        # Page 1 : Rapport Prescriptif de l'IA
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "1. ARTIFICIAL INTELLIGENCE PRESCRIPTIVE EVALUATION", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.multi_cell(0, 5.5, summary_text_cleaned)
        
        # Page 2 : Classement Étudiants
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "2. TOP 5 PERFORMING INTERNS (COMPETITIVE LEADERBOARD)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        col_widths = [45, 55, 20, 25, 45]
        headers = ["Full Name", "Email Address", "Level", "Points (XP)", "Profile"]
        pdf.set_fill_color(243, 244, 246)
        pdf.set_font("Helvetica", "B", 9.5)
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 7.5, h, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for _, row in df_users.head(5).iterrows():
            fullname = clean_pdf_text(f"{row.get('firstName', '')} {row.get('lastName', '')}")[:22]
            email = clean_pdf_text(str(row.get('email', '')))[:26]
            pdf.cell(col_widths[0], 7, fullname, border=1)
            pdf.cell(col_widths[1], 7, email, border=1)
            pdf.cell(col_widths[2], 7, clean_pdf_text(str(row.get('level', '-'))), border=1, align="C")
            pdf.cell(col_widths[3], 7, clean_pdf_text(str(row.get('xp', '0'))), border=1, align="R")
            pdf.cell(col_widths[4], 7, clean_pdf_text(str(row.get('aiProfile', '-'))), border=1, align="C")
            pdf.ln()
        pdf.ln(4)
        pdf.image(user_chart_path, x=15, w=180)

        # Page 3 : Métriques des Quêtes
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "3. QUEST ENGAGEMENT AND COMPLETION METRICS", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        col_widths_quests = [85, 35, 35, 35]
        headers_quests = ["Quest Title", "Completion Rate", "Engagement Rate", "Total Attempts"]
        pdf.set_fill_color(230, 242, 254)
        pdf.set_font("Helvetica", "B", 9.5)
        for i, h_q in enumerate(headers_quests):
            pdf.cell(col_widths_quests[i], 7.5, h_q, border=1, align="C", fill=True)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        if not df_top_quests.empty:
            for _, row in df_top_quests.iterrows():
                q_name = clean_pdf_text(str(row['quest_title']))[:42]
                pdf.cell(col_widths_quests[0], 7, q_name, border=1)
                pdf.cell(col_widths_quests[1], 7, f"{row['completion_rate']:.1f}%", border=1, align="C")
                pdf.cell(col_widths_quests[2], 7, f"{row['engagement_rate']:.1f}%", border=1, align="C")
                pdf.cell(col_widths_quests[3], 7, str(int(row['total_attempts'])), border=1, align="C")
                pdf.ln()
        pdf.ln(4)
        pdf.image(quest_chart_path, x=15, w=180)

        # Page 4 : Badges hebdomadaires dynamiques
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "4. WEEKLY UNLOCKED ACHIEVEMENTS", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)
        if achievement_winners:
            for ach_name, students in achievement_winners.items():
                students_list_str = ", ".join(sorted(list(students)))
                pdf.set_fill_color(241, 245, 249)
                pdf.set_font("Helvetica", "B", 9.5)
                pdf.cell(55, 8, f"  {clean_pdf_text(ach_name)}", border=1, fill=True)
                pdf.set_font("Helvetica", "", 9.5)
                pdf.cell(130, 8, f" obtained by [ {clean_pdf_text(students_list_str)} ]", border=1)
                pdf.ln(10.5)
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, "No achievements were fully unlocked during this tracking period.", new_x="LMARGIN", new_y="NEXT")

        # Sauvegarde en mémoire tampon
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)

        # Suppression des résidus locaux d'images
        for p in [user_chart_path, quest_chart_path]:
            if os.path.exists(p): os.remove(p)

        # Routage sortant SMTP
        email_sender = os.getenv("EMAIL_SENDER")
        email_password = os.getenv("EMAIL_PASSWORD")
        email_receiver = os.getenv("ADMIN_EMAIL")

        message = MIMEMultipart()
        message['From'] = email_sender
        message['To'] = email_receiver
        message['Subject'] = f"🎓 [Talentyz Performance] Extended Visual Analytics Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f4f4f7;">
            <h2>Weekly Talentyz Platform Indicators</h2>
            <p>Active Tracked Students: {len(df_users)}</p>
            <p>L'analyse prescriptive globale basée sur l'IA et l'évaluation graphique comparative sont incluses dans la pièce jointe.</p>
        </body>
        </html>
        """
        message.attach(MIMEText(html, 'html'))
        attachment = MIMEApplication(pdf_buffer.read(), _subtype='pdf')
        attachment.add_header('Content-Disposition', 'attachment', filename=f"Talentyz_Visual_Report_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.pdf")
        message.attach(attachment)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_sender, email_password)
            server.send_message(message)
        print("[+] Rapport d'analyse d'engagement Talentyz expédié avec succès.")
    except Exception as e:
        print(f"[-] Erreur critique lors de la génération automatique du rapport en tâche de fond : {e}")


# --- GESTION DES CYCLE DE VIE DE L'APPLICATION (LIFESPAN SÉCURISÉ) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialisation du scheduler explicitement en UTC
    scheduler = BackgroundScheduler(timezone="UTC")
    
    # 2. Ajout du job hebdomadaire avec datetime conscient du fuseau horaire (Aware)
    # 'next_run_time' mis à 'now' force l'exécution immédiate au démarrage
    scheduler.add_job(
        generate_report_and_send_email, 
        'interval', 
        weeks=1, 
        next_run_time=datetime.now(timezone.utc)
    )
    
    # 3. Démarrage du scheduler
    scheduler.start()
    print("[+] Scheduler démarré en UTC : Premier rapport initié en tâche de fond.")
    
    yield
    
    # 4. Arrêt propre lors de la fermeture de l'application
    scheduler.shutdown()
    print("[-] Scheduler arrêté proprement.")


# --- DÉCLARATION FASTAPI & MIDDLEWARES ---
app = FastAPI(title="JeelQuest Questy V1", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://beta.jeelquest.space"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- ROUTINES COMMUNE DE NETTOYAGE (QUESTY RAG) ---
def get_db_connection():
    if not MONGO_URI:
        raise Exception("MONGO_URI non configurée dans l'environnement.")
    client = MongoClient(MONGO_URI)
    return client["documents_db"]

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = re.sub(r"[¢©®«#]", "", text)
    text = re.sub(r"\b[eJo]\b", "", text)
    text = re.sub(r"^\s*o\s+", "- ", text, flags=re.MULTILINE)
    text = text.replace("&", "and")
    text = re.sub(r'(?<!\S)@(?!\S)', 'at', text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = "\n".join([line.strip() for line in text.splitlines()])
    return text.strip()

def clean_filename(filename):
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")
    filename = filename.replace(" ", "_")
    return re.sub(r"[^\w\.-]", "", filename)

def split_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# --- ENDPOINT 1 : CHARGEMENT ET CHUNKING DES DOCUMENTS ---
@app.post("/upload-documents/")
async def upload_files(document: UploadFile = File(...)):
    uploaded_files = []
    db = get_db_connection()
    documents_collection = db["documents"]

    global vectorstore
    if vectorstore is None:
        vectorstore = setup_vectorstore()

    if not document.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les documents au format PDF sont autorisés.")

    file_path = None
    try:
        safe_filename = clean_filename(document.filename)
        file_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, safe_filename))

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(document.file, buffer)

        if not os.path.exists(file_path):
            raise Exception(f"Fichier mal enregistré sur le disque local : {file_path}")

        print(f"[+] Document enregistré temporairement : {file_path}")

        doc = fitz.open(file_path)
        raw_text = ""
        for page in doc:
            raw_text += page.get_text("text") + "\n"
        doc.close()

        extracted_text = clean_text(raw_text)
        chunks = split_text(extracted_text)
        print(f"[+] Nombre de segments générés : {len(chunks)}")

        docs_langchain = [
            Document(page_content=chunk, metadata={"source": document.filename}) 
            for chunk in chunks if chunk.strip()
        ]

        if docs_langchain:
            print(f"[+] Indexation de {len(docs_langchain)} fragments dans la base vectorielle...")
            vectorstore.add_documents(docs_langchain)
            print(" Synchronisation de l'index vectoriel terminée.")
        else:
            print(" Aucun fragment de texte exploitable détecté.")

        doc_entry = {
            "filename": document.filename,
            "filepath": file_path,
            "content": extracted_text,
            "chunks": chunks,
            "created_at": datetime.now(timezone.utc)
        }
        documents_collection.insert_one(doc_entry)
        uploaded_files.append(document.filename)

        return {
            "message": "Fichier uploadé, segmenté et indexé avec succès.",
            "files": uploaded_files
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


# --- ENDPOINT 2 : CHATBOT INTELLIGENT (QUESTY INFERENCE) ---
def get_retriever():
    return vectorstore.as_retriever(search_kwargs={"k": 4})

class ChatRequest(BaseModel):
    query: str

@app.post("/chatbot/")
async def chatbot(request: ChatRequest):
    try:
        db = get_db_connection()
        chat_collection = db["chat_history"]

        # Utilisation de GOOGLE_API_KEY (Clé dédiée pour Questy)
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )

        retriever = get_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": request.query})
        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        sources_text = [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source", "unknown")
            }
            for doc in source_docs
        ]

        chat_entry = {
            "query": request.query,
            "answer": answer,
            "num_sources": len(source_docs),
            "sources": sources_text,
            "created_at": datetime.now(timezone.utc)
        }
        chat_collection.insert_one(chat_entry)

        return {
            "query": request.query,
            "answer": answer,
            "sources": sources_text
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
