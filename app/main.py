import os
import io
import re
import shutil
import unicodedata
import requests
import base64
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
        # Configuration du bandeau supérieur gris foncé
        self.set_fill_color(31, 41, 55) # Couleur Ardoise / Gris foncé
        self.rect(0, 0, 210, 35, "F") # Dessine le rectangle plein sur toute la largeur (A4: 210mm)
        
        # Configuration du texte dans l'en-tête
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 14)
        self.set_y(10)
        # Titre principal aligné à gauche
        self.cell(0, 10, "TALENTYZ - BUSINESS ANALYTICS & ENGAGEMENT REPORT", new_x="RIGHT", new_y="TOP", align="L")
        
        # Date du jour alignée à droite
        self.set_font("Helvetica", "", 10)
        self.cell(0, 10, f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", new_x="LMARGIN", new_y="NEXT", align="R")
        self.ln(20) # Saut de ligne pour espacer le contenu du document

    def footer(self):
        # Positionnement à 15 mm du bas de la page
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(156, 163, 175) # Couleur grise claire pour la discrétion
        # Numérotation dynamique de la page et mention de confidentialité
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
    You MUST strictly adhere to this exact output format for headers, with a double line break after each section title:

    1. GLOBAL ENGAGEMENT DIAGNOSTIC
    [Insert the engagement diagnosis paragraph here]

    2. QUEST PERFORMANCE AND ANALYSIS
    [Insert the quest analysis paragraph here]

    CRITICAL INSTRUCTIONS: 
    1. Do NOT use any Markdown formatting (*, #, etc.).
    2. Do NOT write headers or whole paragraphs in block CAPITAL LETTERS.
    3. You must ensure there is a clear, empty line separating the section titles (1. GLOBAL...) from the start of their corresponding paragraphs. This is for PDF readability.
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
        print("[+] Étape 1 : Début du cycle analytique...")
        print(f"[+] Étape 2 : Tentative de connexion à MongoDB avec URI...")
        client_mongo = MongoClient(MONGO_URI_1 if MONGO_URI_1 else MONGO_URI)
        db = client_mongo["pfe"]

        obj_university_id = ObjectId(TARGET_UNIVERSITY_ID) if ObjectId.is_valid(TARGET_UNIVERSITY_ID) else None
        query_user = {"$or": [{"universityId": TARGET_UNIVERSITY_ID}, {"universityId": obj_university_id}]}
        
        print("[+] Étape 3 : Requête de récupération des utilisateurs...")
        users_list = list(db["users"].find(query_user))
        print(f"[+] Résultat de la requête : {len(users_list)} utilisateurs trouvés.")
        
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
        total_students = len(df_users)
        avg_xp = df_users['xp'].mean() if 'xp' in df_users.columns else 0

        summary_text = generate_ai_academic_summary(df_users, quests_summary_str, total_available_quests)
        summary_text_cleaned = clean_pdf_text(summary_text)

        # === GRAPHIQUE 1 : CLASSEMENT HORIZONTAL MIS À JOUR (IMAGE 3) ===
        sns.set_theme(style="whitegrid")
        df_top5_users = df_users.head(5).copy()
        df_top5_users['clean_name'] = df_top5_users.apply(lambda r: clean_pdf_text(f"{r.get('firstName', '')} {r.get('lastName', '')}"), axis=1)
        
        fig, ax1 = plt.subplots(figsize=(7.5, 3.8))
        bars_u = ax1.barh(df_top5_users['clean_name'], df_top5_users['xp'], color='#4f46e5', height=0.5, edgecolor='none')
        ax1.invert_yaxis()  # Inverse l'axe pour afficher le premier en haut
        ax1.set_title("GENERAL Intern EXPERIENCE LEADERBOARD (XP)", fontsize=12, fontweight='bold', color='#1f2937', pad=15)
        ax1.set_xlabel("Experience Points (XP)", fontsize=10, labelpad=8)
        ax1.tick_params(axis='both', labelsize=10)
        
        for bar in bars_u:
            width = bar.get_width()
            ax1.text(width + (width * 0.01 + 2), bar.get_y() + bar.get_height()/2, f'{int(width)} XP', 
                     va='center', ha='left', fontsize=10, fontweight='bold', color='#374151')
                     
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        user_chart_path = "temp_user_xp.png"
        fig.savefig(user_chart_path, dpi=250, format='png')
        plt.close(fig)

        # === GRAPHIQUE 2 : ANALYSE DES QUÊTES AVEC ANNOTATIONS (IMAGE 5) ===
        fig, ax2 = plt.subplots(figsize=(7.5, 3.8))
        df_top5_q_chart = df_top_quests.copy() if not df_top_quests.empty else pd.DataFrame(columns=['quest_title', 'engagement_rate', 'completion_rate'])
        
        if not df_top5_q_chart.empty:
            df_top5_q_chart['clean_title'] = df_top5_q_chart['quest_title'].apply(lambda x: clean_pdf_text(x)[:26] + "..." if len(clean_pdf_text(x)) > 28 else clean_pdf_text(x))
            bars_q = ax2.bar(df_top5_q_chart['clean_title'], df_top5_q_chart['engagement_rate'], color='#0284c7', alpha=0.8, width=0.35, label="Engagement Rate")
            ax2.plot(df_top5_q_chart['clean_title'], df_top5_q_chart['completion_rate'], color='#10b981', marker='o', markersize=6, linewidth=2.5, label="Completion Rate")
            
            for i, txt in enumerate(df_top5_q_chart['completion_rate']):
                ax2.annotate(f"{txt:.1f}%", (df_top5_q_chart['clean_title'].iloc[i], df_top5_q_chart['completion_rate'].iloc[i]),
                             textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, fontweight='bold', color='#047857')

        ax2.set_ylabel("Percentage (%)", fontsize=10, labelpad=8)
        ax2.set_ylim(0, 115)
        ax2.set_title("COMPARATIVE QUEST IMPACT AND COMPLETION METRICS", fontsize=12, fontweight='bold', color='#1f2937', pad=15)
        ax2.set_xticklabels(df_top5_q_chart['clean_title'] if not df_top5_q_chart.empty else [], rotation=15, ha='right', fontsize=9)
        ax2.tick_params(axis='y', labelsize=10)
        ax2.legend(loc="upper right", fontsize=9, frameon=True, facecolor='#ffffff', edgecolor='#e5e7eb')
        sns.despine()
        plt.tight_layout()
        quest_chart_path = "temp_quest_perf.png"
        fig.savefig(quest_chart_path, dpi=250, format='png')
        plt.close(fig)

        # === 5. CONSTRUCTION DU DOCUMENT PDF DE FAÇON STRUCTURÉE ===
        print("[+] Structuring PDF document pages...")
        pdf = CorporatePDF()
        pdf.alias_nb_pages()
        
        # --- PAGE 1 : ÉVALUATION PRESCRIPTIVE DE L'IA ---
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 8, "1. ARTIFICIAL INTELLIGENCE PRESCRIPTIVE EVALUATION", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        pdf.set_text_color(55, 65, 81)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.multi_cell(0, 5.5, summary_text_cleaned)
        
        # --- PAGE 2 : CLASSEMENT ÉTUDIANTS & GRAPHIQUE ---
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 8, "2. TOP 5 PERFORMING INTERNS (COMPETITIVE LEADERBOARD)", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        
        col_widths = [45, 55, 20, 25, 45] 
        headers = ["Full Name", "Email Address", "Level", "Points (XP)", "Profile"]
        
        pdf.set_fill_color(243, 244, 246)
        pdf.set_text_color(17, 24, 39)
        pdf.set_font("Helvetica", "B", 9.5)
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7.5, header, border=1, align="C", fill=True)
        pdf.ln()
        
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(55, 65, 81)
        for _, row in df_users.head(5).iterrows():
            fullname = clean_pdf_text(f"{row.get('firstName', '')} {row.get('lastName', '')}")
            email = clean_pdf_text(str(row.get('email', '')))
            level = clean_pdf_text(str(row.get('level', '-')))
            xp = clean_pdf_text(str(row.get('xp', '0')))
            ai_profile = clean_pdf_text(str(row.get('aiProfile', 'Not Defined')))
            
            if len(fullname) > 22: fullname = fullname[:20] + "..."
            if len(email) > 26: email = email[:24] + "..."
            
            pdf.cell(col_widths[0], 7, fullname, border=1, align="L")
            pdf.cell(col_widths[1], 7, email, border=1, align="L")
            pdf.cell(col_widths[2], 7, level, border=1, align="C")
            pdf.cell(col_widths[3], 7, xp, border=1, align="R")
            pdf.cell(col_widths[4], 7, ai_profile, border=1, align="C")
            pdf.ln()
            
        pdf.ln(4)
        current_y = pdf.get_y()
        target_y = current_y + 8 if current_y + 85 < 270 else 90
        pdf.image(user_chart_path, x=15, y=target_y, w=180)
        
        # --- PAGE 3 : PERFORMANCES DES QUÊTES & GRAPHIQUE ANALYTIQUE ---
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(31, 41, 55)
        pdf.cell(0, 8, "3. QUEST ENGAGEMENT AND COMPLETION METRICS", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        
        col_widths_quests = [85, 35, 35, 35] 
        headers_quests = ["Quest Title", "Completion Rate", "Engagement Rate", "Total Attempts"]
        
        pdf.set_fill_color(230, 242, 254) 
        pdf.set_text_color(30, 58, 138)
        pdf.set_font("Helvetica", "B", 9.5)
        for i, h_q in enumerate(headers_quests):
            pdf.cell(col_widths_quests[i], 7.5, h_q, border=1, align="C", fill=True)
        pdf.ln()
        
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(55, 65, 81)
        
        if not df_top_quests.empty:
            for _, row in df_top_quests.iterrows():
                q_name = clean_pdf_text(str(row['quest_title']))
                if len(q_name) > 42: q_name = q_name[:40] + "..."
                
                c_rate = f"{row['completion_rate']:.1f}%"
                e_rate = f"{row['engagement_rate']:.1f}%"
                attempts = str(int(row['total_attempts']))
                
                pdf.cell(col_widths_quests[0], 7, q_name, border=1, align="L")
                pdf.cell(col_widths_quests[1], 7, c_rate, border=1, align="C")
                pdf.cell(col_widths_quests[2], 7, e_rate, border=1, align="C")
                pdf.cell(col_widths_quests[3], 7, attempts, border=1, align="C")
                pdf.ln()
        else:
            pdf.cell(sum(col_widths_quests), 8, "No active quest metrics available.", border=1, align="C")
            pdf.ln()
            
        pdf.ln(4)
        current_y_q = pdf.get_y()
        target_y_q = current_y_q + 8 if current_y_q + 85 < 270 else 90
        pdf.image(quest_chart_path, x=15, y=target_y_q, w=180)

        # --- PAGE 4 : BADGES HEBDOMADAIRES DYNAMIQUES ---
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, f"4. WEEKLY UNLOCKED ACHIEVEMENTS", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6) 
        
        if achievement_winners:
            for ach_name, students in achievement_winners.items():
                students_list_str = ", ".join(sorted(list(students)))
                
                pdf.set_fill_color(241, 245, 249) 
                pdf.set_draw_color(226, 232, 240) 
                
                pdf.set_font("Helvetica", "B", 9.5)
                pdf.set_text_color(30, 41, 59) 
                pdf.cell(55, 8, f"  {clean_pdf_text(ach_name)}", border=1, fill=True, new_x="RIGHT", new_y="TOP")
                
                pdf.set_font("Helvetica", "", 9.5)
                pdf.set_text_color(71, 85, 105)
                content_text = f" obtained by [ {clean_pdf_text(students_list_str)} ]"
                
                pdf.cell(130, 8, content_text, border=1, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2.5) 
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(148, 163, 184)
            pdf.cell(0, 8, "No achievements were fully unlocked during this tracking period.", new_x="LMARGIN", new_y="NEXT")
            
        # Sauvegarde en mémoire tampon
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)

        # Suppression immédiate des fichiers graphiques locaux temporaires
        for p in [user_chart_path, quest_chart_path]:
            if os.path.exists(p): os.remove(p)

        # --- ROUTAGE SORTANT VIA L'API DE RESEND (PORT 443 HTTPS) ---
        print("[+] Étape 5 : Préparation de l'envoi via l'API HTTP Resend...")
        resend_api_key = os.getenv("RESEND_API_KEY")
        if not resend_api_key:
            print("[-] Erreur : RESEND_API_KEY manquante dans l'environnement Railway.")
            return

        pdf_base64 = base64.b64encode(pdf_buffer.read()).decode('utf-8')
        filename_report = f"Talentyz_Visual_Report_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.pdf"

        payload = {
            "from": "Talentyz Analytics Agent <onboarding@resend.dev>",  
            "to": ["akremrhaimi@gmail.com"],
            "subject": f"🎓 [Talentyz Performance] Extended Visual Analytics Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            "html": f"""
            <html>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #f3f4f6; padding: 30px; color: #111827; margin: 0;">
        <div style="max-width: 650px; margin: 0 auto; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
            
            <div style="background-color: #1e293b; padding: 30px 20px; text-align: center; color: #ffffff;">
                <h1 style="margin: 0; font-size: 24px; font-weight: bold; letter-spacing: 0.5px;">Talentyz Intern Activity Report</h1>
                <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 14px;">Predictive AI Analysis & Goal Monitoring</p>
            </div>
            
            <div style="padding: 30px 25px; line-height: 1.6;">
                <p style="font-size: 15px; margin-top: 0;">Hello,</p>
                <p style="font-size: 14px; color: #334155;">Platform performance indicators for <strong>Talentyz</strong> University have been successfully consolidated.</p>
                
                <div style="background-color: #f8fafc; border-left: 4px solid #4f46e5; padding: 20px; margin: 25px 0; border-radius: 0 8px 8px 0;">
                    <h3 style="margin-top: 0; margin-bottom: 15px; color: #1e293b; font-size: 15px;">
                        <span style="margin-right: 8px;">📊</span> Core Metrics Preview:
                    </h3>
                    
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                        <tr>
                            <td style="padding: 6px 0; color: #475569;">Active Tracked Interns:</td>
                            <td style="padding: 6px 0; text-align: right; font-weight: bold; color: #1e293b;">{total_students}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; color: #475569;">Average Experience Points:</td>
                            <td style="padding: 6px 0; text-align: right; font-weight: bold; color: #1e293b;">{avg_xp:.1f} XP</td>
                        </tr>
                    </table>
                </div>
                
                <p style="font-size: 14px; color: #334155; margin-bottom: 30px;">
                    The attached PDF document features the comprehensive AI prescriptive analysis on learning dynamics directly on the first page, followed by granular rankings and comparative graphical evaluations (Quests, Levels, Profiles).
                </p>
                
                <hr style="border: 0; border-top: 1px solid #e2e8f0; margin-bottom: 20px;">
                
                <div style="text-align: center; color: #94a3b8; font-size: 12px;">
                    Automated Transmission - Questy AI Academic Reporting Engine.
                </div>
            </div>
        </div>
    </body>
    </html>
            """,
            "attachments": [
                {
                    "content": pdf_base64,
                    "filename": filename_report
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {resend_api_key}",
            "Content-Type": "application/json"
        }

        print("[+] Envoi de la requête HTTP POST à l'API Resend (Port 443)...")
        response = requests.post("https://api.resend.com/emails", json=payload, headers=headers)

        if response.status_code in [200, 201]:
            print("[+] Rapport d'analyse d'engagement Talentyz expédié avec succès via l'API Resend ! ✨")
        else:
            print(f"[-] Échec de l'envoi via Resend. Code : {response.status_code}, Réponse : {response.text}")
    except Exception as e:
        print(f"[-] Erreur critique lors de la génération automatique du rapport en tâche de fond : {e}")

# --- DÉCLARATION FASTAPI & MIDDLEWARES ---
app = FastAPI(title="JeelQuest Questy V1", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://beta.jeelquest.space"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- GESTION DES CYCLE DE VIE DE L'APPLICATION (LIFESPAN CORRIGÉ) ---
@app.on_event("startup")
async def startup_event():
    print("[+] Initialisation obligatoire du Vectorstore pour Questy RAG...")
    setup_vectorstore()

    global scheduler
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        generate_report_and_send_email, 
        'interval', 
        weeks=1
    )
    scheduler.start()
    print("[+] Scheduler démarré en UTC.")

    # Lancement initial asynchrone sécurisé en tâche de fond
    asyncio.create_task(asyncio.to_thread(generate_report_and_send_email))
    print("[+] Premier rapport forcé au démarrage dans un thread dédié.")

@app.on_event("shutdown")
def shutdown_event():
    global scheduler
    if 'scheduler' in globals():
        scheduler.shutdown()
        print("[-] Scheduler arrêté proprement.")

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
    text = re.sub(r"^\so\s+", "- ", text, flags=re.MULTILINE)
    text = text.replace("&", "and")
    text = re.sub(r'(?<!\S)@(?!\S)', 'at', text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"-\s\n\s*", "", text)
    text = "\n".join([line.strip() for line in text.splitlines()])
    return text.strip()

def clean_filename(filename):
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")
    filename = filename.replace(" ", "_")
    return re.sub(r"[^\w.-]", "", filename)

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
    global vectorstore
    if vectorstore is None:
        setup_vectorstore()
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore non initialisé.")
    return vectorstore.as_retriever(search_kwargs={"k": 4})

class ChatRequest(BaseModel):
    query: str

@app.post("/chatbot/")
async def chatbot(request: ChatRequest):
    try:
        db = get_db_connection()
        chat_collection = db["chat_history"]

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
