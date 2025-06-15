# 📚 Automated Book Publication Workflow using AI Agents

This project implements an **agentic AI-powered system** that fetches literary content from a web URL, rewrites it in a modern storytelling style using LLMs, allows human review and editing, and stores every version with intelligent search and retrieval.

## 🚀 Features

### ✅ Key Capabilities
1. **Scraping & Screenshot**  
   Scrapes chapters from literary sources like Wikisource and captures full-page screenshots using **Playwright**.

2. **AI Writing & Review**  
   Uses **Gemini 1.5 Flash** to rewrite old-style literature into modern, vivid storytelling while preserving meaning and emotional clarity.

3. **Human-in-the-Loop Editing**  
   Allows users to accept or manually refine AI-spun chapters. Edits are saved and versioned.

4. **Agentic API Design**  
   Modular agent-based structure for Scraper, AI Writer, Editor, Versioner, and Retriever for seamless flow.

5. **Versioning & Consistency with ChromaDB**  
   Stores every final/edited version with metadata. Enables semantic search and intelligent retrieval using RL-style ranking.

---

## 🧠 Core Tools & Technologies

| Tool         | Purpose                                |
|--------------|----------------------------------------|
| Python       | Core development                       |
| Playwright   | Web scraping and screenshot capture    |
| Gemini 1.5  Flash | AI rewriting (text spinning)           |
| ChromaDB     | Document versioning and storage        |
| Terminal UI  | Human-in-the-loop CLI editor           |

---

## 🛠 How It Works

### Clone the github repository
```bash
git clone https://github.com/yourusername/book-pipeline-agents.git
cd book-pipeline-agents
```

### Install dependencies
```bash
pip install -r requirements.txt
OR
pip install playwright bs4 google-generativeai dotenv chromadb datetime numpy
```

### Add your .env
```bash
GEMINI_API_KEY=your_google_api_key
```
### 🔧 Run the Full Agentic Pipeline
```bash
python main.py
```

