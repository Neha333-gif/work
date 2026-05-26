# 4. AI Research Assistant (RAG + Agents)

# Agent that:

# Scrapes papers
# Summarizes findings
# Generates insights
# ➡️ Useful for students + shows tool use + retrieval + reasoning

import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import random
import os
from datetime import datetime
from collections import defaultdict
import json
import requests
import numpy as np 
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as PydanticBaseModel
import threading
import queue

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state for logs
log_queue = queue.Queue()

class CrewOutputHandler:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout

    def write(self, data):
        if data.strip():
            # Filter out noisy uvicorn request logs from the live UI
            if "GET /logs" not in data and "GET /status" not in data and "INFO:" not in data:
                log_queue.put(data.strip())
        try:
            self.original_stdout.write(data)
        except UnicodeEncodeError:
            self.original_stdout.write(data.encode('ascii', 'replace').decode('ascii'))

    def flush(self):
        self.original_stdout.flush()

    def isatty(self):
        return self.original_stdout.isatty()

# Custom print handler to capture crewai output while keeping console output
import sys
sys.stdout = CrewOutputHandler(sys.stdout)

my_llm = LLM(model="groq/llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0.7, max_tokens=1024)

vector_store = {"papers": []}
relavant_papers = []
simmilarity_scores = []
best_papers = []
model = SentenceTransformer('all-MiniLM-L6-v2')

class FindPapersInput(BaseModel):
    query: str = Field(..., description="The research topic or keywords to search for.")
    max_results: int = Field(10, description="The number of papers to retrieve (maximum 10 to avoid rate limits).")

class find_papers(BaseTool):
    name : str = "scraper"
    description : str = "Scrapes research papers from Arxiv."
    args_schema: Type[BaseModel] = FindPapersInput
    
    def _run(self, query, max_results):
        max_results = min(int(max_results), 10) # Clamp to 10 for rate limit safety
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
        res = requests.get(url)
        if res.status_code != 200:
            return f"Error: Arxiv API returned status code {res.status_code}"

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(res.text)
        except ET.ParseError:
            return "Error: Could not parse Arxiv API response."

        papers = []
        for entry in root.findall("atom:entry", ns):
            papers.append({
                "title" : entry.find("atom:title", ns).text,
                "summary" : entry.find("atom:summary", ns).text,
                "link" : entry.find("atom:link", ns).attrib.get('href') if entry.find("atom:link", ns) is not None else "",
                "author_name" : [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]})
        vector_store["papers"].extend(papers)
        return f"Successfully found and stored {len(papers)} papers in the vector database."


class SearchVectorDBInput(BaseModel):
    query: str = Field(..., description="The search query for the vector database.")

class search_vectordb(BaseTool):
    name : str = "searcher"
    description : str = "Search for relevant papers in the database."
    args_schema: Type[BaseModel] = SearchVectorDBInput

    def _run(self, query: str) -> str:
        query = query.lower()
        docs = [p["summary"] for p in vector_store["papers"]]

        query_dim = model.encode(query)
        
        
        for paper in vector_store["papers"]:
            emb_doc = model.encode(paper["summary"])
            similarity = np.dot(query_dim, emb_doc)
            simmilarity_scores.append({"paper": paper, "similarity_score": similarity})

        simmilarity_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
        sorted_scores = simmilarity_scores[:3]

        best_papers_list = [{"index": i+1, "title": p["paper"]["title"], "summary": p["paper"]["summary"]} 
                            for i, p in enumerate(sorted_scores)]
        return json.dumps(best_papers_list)
        

class SaveSummaryInput(BaseModel):
    summary: str = Field(..., description="The summary of the research papers.")

class save_summary(BaseTool):
    name : str = "save_summary"
    description : str = "Save the generated summary to the system."
    args_schema: Type[BaseModel] = SaveSummaryInput

    def _run(self, summary: str) -> str:
        saved_summary = []
        saved_summary.append({
            "id" : f"{len(vector_store['papers'])+1}",
            "content" : summary,
            "date_time" : datetime.now().strftime("%y:%m:%d"),
            "len_summary" : len(summary.split())})
        return json.dumps(saved_summary)
    

class SaveInsightsInput(BaseModel):
    insights: str = Field(..., description="The generated insights from the research.")

class save_insights(BaseTool):
    name : str = "save_insights"
    description : str = "Save the generated insights to the system."
    args_schema: Type[BaseModel] = SaveInsightsInput

    def _run(self, insights: str) -> str: 
        saved_insight = []
        saved_insight.append({
            "id" : "1",
            "insight" : insights,
            "datetime" : datetime.now().strftime("%y:%m:%d"),
            "len_insight" : len(insights.split()) })
        return json.dumps(saved_insight)

class RankCitationsInput(BaseModel):
    query: str = Field(..., description="The query to rank papers by citations (dummy input for consistency).")

class rank_citations(BaseTool):
    name : str = "ranker"
    description : str = "Rank papers based on citations."
    args_schema: Type[BaseModel] = RankCitationsInput

    def _run(self, query: str) -> str:
        if not vector_store["papers"]:
            return "No papers found."
        sorted_papers = sorted(vector_store["papers"], key=lambda x: x.get("citations", 0), reverse=True)
        return json.dumps(sorted_papers[:5])

find_papers_tool = find_papers()
search_vectordb_tool = search_vectordb()
save_summary_tool = save_summary()
save_insights_tool = save_insights()


researcher = Agent(
    role = "researcher",
    goal = "to research and find for best papers based on topic",
    backstory = "expert in finding best papers based on topic",
    llm = my_llm,
    tools = [find_papers_tool, search_vectordb_tool, rank_citations()],
    allow_delegation = False, verbose = True)


writer = Agent(
    role = "writer",
    goal = "to write and publish the research paper",
    backstory = "expert in writing and publishing research papers",
    llm = my_llm,
    tools = [save_summary_tool],
    allow_delegation = False, verbose = True)

generate_insights = Agent(
    role = "generate_insights",
    goal = "to generate insights based on the research",
    backstory = "expert in generating insights based on research",
    llm = my_llm,
    tools = [save_insights_tool],
    allow_delegation = False, verbose = True)

# tasks 

Task_1 = Task(
    description = "research on recent trends in AI for healthcare",
    agent = researcher,
    expected_output = "A list of papers based on topic")

Task_2 = Task(
    description = "write a summary of papers based on topic",
    agent = writer,
    expected_output = "A summary of papers based on topic")

Task_3 = Task(
    description = "generate insights based on the research",
    agent = generate_insights,
    expected_output = "Insights based on the research")


# FastAPI Endpoints
class ResearchRequest(PydanticBaseModel):
    topic: str

@app.post("/run-research")
async def run_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    def execution_wrapper():
        try:
            log_queue.put("SYSTEM: Starting research workflow...")
            # Update task description with user topic
            Task_1.description = f"research on recent trends in {request.topic}"
            Task_2.description = f"write a summary of papers based on {request.topic}"
            Task_3.description = f"generate insights based on the research for {request.topic}"
            
            crew = Crew(
                agents = [researcher, writer, generate_insights],
                tasks = [Task_1, Task_2, Task_3],
                verbose = True,
                max_rpm = 29)
            
            result = crew.kickoff()
            log_queue.put(f"FINAL_RESULT: {result}")
        except Exception as e:
            log_queue.put(f"SYSTEM_ERROR: {str(e)}")
            import traceback
            log_queue.put(traceback.format_exc())

    threading.Thread(target=execution_wrapper).start()
    return {"status": "started"}

@app.get("/logs")
async def get_logs():
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return {"logs": logs}

@app.get("/papers")
async def get_papers():
    return {"papers": vector_store["papers"]}

# User Management
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

class UserAuth(PydanticBaseModel):
    username: str
    password: str

@app.post("/register")
async def register(user: UserAuth):
    users = load_users()
    if user.username in users:
        return {"status": "error", "message": "User already exists"}
    users[user.username] = user.password
    save_users(users)
    return {"status": "success"}

@app.post("/login")
async def login(user: UserAuth):
    users = load_users()
    if user.username in users and users[user.username] == user.password:
        return {"status": "success"}
    return {"status": "error", "message": "Invalid credentials"}

# Serve frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
