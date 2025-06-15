#main.py

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import os
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
from datetime import datetime
import numpy as np

# === AGENT: Scraper ===
def scrape_chapter(url: str, text_path: str = "scraped_text.txt", screenshot_path: str = "screenshot.png") -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.screenshot(path=screenshot_path, full_page=True)

        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        prp_div = soup.find('div', class_='prp-pages-output')

        if prp_div:
            paragraphs = prp_div.find_all('p')
            with open(text_path, "w", encoding="utf-8") as file:
                for para in paragraphs:
                    text = para.get_text(strip=True)
                    file.write(text + "\n")
            return text_path
        else:
            raise ValueError("No <div class='prp-pages-output'> found.")

# === AGENT: AI Writer ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def ai_spin_text(input_path: str, output_path: str = "spun_text.txt") -> str:
    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping AI spin.")
        return output_path

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    response = model.generate_content([
        "You are an AI writer. Rewrite the following literary text in a modern, vivid storytelling style. "
        "Keep the original meaning but improve its emotional and narrative clarity.",
        f"Chapter Content:{content}"
    ])
    spun_text = response.text.strip()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(spun_text)
    return output_path

# === AGENT: Versioning and Storage ===
client = chromadb.Client()
collection = client.get_or_create_collection("chapter_versions")

def save_to_chromadb(text: str, filename: str):
    version_id = filename.replace(".txt", "")
    collection.add(
        documents=[text],
        metadatas=[{"filename": filename, "timestamp": datetime.now().isoformat()}],
        ids=[version_id]
    )

def rl_rank_documents(documents, query):
    def cosine_sim(a, b):
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        if not a_set or not b_set:
            return 0.0
        intersection = len(a_set.intersection(b_set))
        return intersection / ((len(a_set) * len(b_set)) ** 0.5)

    scored = [(doc, cosine_sim(query, doc)) for doc in documents]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0] if scored else "No relevant versions found."

def search_versions(query_text: str, n_results: int = 3):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return rl_rank_documents(results['documents'][0], query_text) if results['documents'] else "No relevant versions found."

# === AGENT: Terminal-based Human-in-the-loop Review ===
def human_review(path_in="spun_text.txt", path_out=None):
    print("\n --- AI Rewritten Chapter ---\n")
    with open(path_in, "r", encoding="utf-8") as f:
        content = f.read()

    print(content)
    print("\nDo you want to:\n1. Accept as-is\n2. Edit manually\n")
    choice = input("Enter choice (1/2): ").strip()

    if not path_out:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_out = f"edited_{timestamp}.txt"

    if choice == "1":
        with open(path_out, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"\n Saved as final version: {path_out}")
        save_to_chromadb(content, path_out)
    elif choice == "2":
        print("\n Enter your edited content below (end with a blank line):\n")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        edited = "\n".join(lines)
        with open(path_out, "w", encoding="utf-8") as f:
            f.write(edited)
        print(f"\n Your edited version saved to: {path_out}")
        save_to_chromadb(edited, path_out)
    else:
        print(" Invalid choice.")

# === AGENT: Pipeline Runner ===
def run_agentic_pipeline(url: str):
    print("[1] Scraping...")
    scraped_path = scrape_chapter(url)

    print("[2] Spinning with Gemini...")
    spun_path = ai_spin_text(scraped_path)

    print("[3] Storing version in ChromaDB...")
    with open(spun_path, "r", encoding="utf-8") as f:
        spun_text = f.read()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(spun_text)
    save_to_chromadb(spun_text, filename)
    print(f" Finished. Saved as {filename}")

if __name__ == "__main__":
    run_agentic_pipeline("https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1")
    while True:
        print("\n--- Agentic Pipeline Completed! ---")
        print("1. For Editing")
        print("2. For RL Search")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            print("Opening human review editor...")
            human_review()
        elif choice == "2":
            keyword = input("Enter your keyword (Ex: chief,betrayal) ").strip()
            print("Searching previous versions...")
            print(search_versions(keyword))
        elif choice == "3":
            print("Exiting the program. Goodbye!")
            break
        else:
            print(" Invalid choice, please try again.")
