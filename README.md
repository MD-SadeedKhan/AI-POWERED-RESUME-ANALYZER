# 🚀 AI-Powered Resume Analyzer  

![Python](https://img.shields.io/badge/Python-3.11-blue)  
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green)  
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue)  
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-orange)  
![License](https://img.shields.io/badge/License-MIT-yellow)  

---

## 🌟 Project Overview  

### The Challenge  
In today's competitive landscape, manual resume screening is a **time-consuming, labor-intensive, and biased** process that slows down hiring efficiency. Recruiters spend countless hours sifting through applications, struggling to identify the best-fit candidates quickly and objectively.  

### The Solution  
The **AI-Powered Resume Analyzer** automates resume screening using **Machine Learning (ML) & NLP**. It goes beyond keyword matching by understanding the **semantic context** of skills and experience, ensuring accurate and actionable insights for recruiters.  

---

## ✨ Key Features & Capabilities  
- ⚡ **Intelligent ATS Scoring** → Weighted scoring (60% skill-match + 40% semantic similarity).  
- 🧠 **Advanced Skill Extraction** → Extracts Technical, Cloud, AI/ML, DevOps, and Soft Skills.  
- 🔍 **Semantic Embeddings** → SentenceTransformers (`all-MiniLM-L6-v2`) for deep semantic understanding.  
- 🚀 **High-Speed Vector Search** → Powered by **FAISS** for lightning-fast similarity checks.  
- 🌐 **RESTful API** → Built with **FastAPI**, clean and fully documented.  
- 💾 **Persistent Storage** → PostgreSQL for secure, structured data.  
- 🛠 **Modular Architecture** → Scalable, maintainable, and enterprise-ready.  

---

## 🛠 Tech Stack  

| Layer            | Technology                           | Description                                                                 |
|------------------|--------------------------------------|-----------------------------------------------------------------------------|
| **Backend**      | Python 3.11, FastAPI                 | High-performance async web framework                                        |
| **ML/NLP**       | SentenceTransformers, scikit-learn   | Embeddings, skill extraction, similarity computations                      |
| **Vector Store** | FAISS                                | Efficient similarity search on large-scale embeddings                       |
| **Database**     | PostgreSQL                           | Secure, relational storage                                                  |
| **Validation**   | Pydantic                             | Data integrity & type safety for APIs                                       |
| **Testing**      | Pytest                               | Automated testing suite                                                     |
| **Deployment**   | Render, Vercel, AWS                  | Flexible hosting options                                                    |
| **Logging**      | Loguru                               | Simple & powerful logging                                                   |

---

## 🏛 System Architecture & Data Flow  

1. **Resume Upload** → User uploads resume via `/analyze` endpoint.  
2. **Data Ingestion** → Resume text + metadata stored in PostgreSQL.  
3. **Semantic Processing** → Embeddings generated using SentenceTransformers.  
4. **Vector Indexing** → Embeddings stored in FAISS for fast retrieval.  
5. **Score Calculation** → Combines semantic similarity + skill matching → ATS Score.  
6. **Response Generation** → JSON response with ATS score, matched & missing skills.  

📌 **Diagram (placeholder – add your architecture image here):**  
![System Architecture](assets/architecture.png)  

---

## 📂 Folder Structure  

```bash
.
├── database/            # SQLAlchemy engine, session, and database setup
├── models/              # Pydantic and SQLAlchemy ORM models
├── routes/              # FastAPI API endpoints for different features
├── vectorstore/         # FAISS vector index management
├── tests/               # Unit, integration, and end-to-end tests
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
└── README.md            # Documentation

⚙️ Setup & Installation
1️⃣ Clone the Repository

git clone https://github.com/your-username/AI-POWERED-RESUME-ANALYZER.git
cd AI-POWERED-RESUME-ANALYZER

2️⃣ Create & Activate Conda Environment

conda create -n resume-analyzer python=3.11 -y
conda activate resume-analyzer

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Setup PostgreSQL Database

Create a database in PostgreSQL (e.g., resume_db).

Update your .env file with database credentials:

DATABASE_URL=postgresql://username:password@localhost:5432/resume_db

5️⃣ Run Database Migrations

alembic upgrade head

6️⃣ Launch FastAPI Serveruvicorn main:app --reload

uvicorn main:app --reload

Server will be available at: 👉 http://127.0.0.1:8000

Interactive API Docs: 👉 http://127.0.0.1:8000/docs

🧪 Testing

pytest -v


🚀 Deployment

Render: One-click FastAPI + PostgreSQL hosting.

Vercel: Deploy frontend & connect with backend API.

AWS/GCP/Azure: For enterprise-scale deployment.

📜 License

This project is licensed under the MIT License.

