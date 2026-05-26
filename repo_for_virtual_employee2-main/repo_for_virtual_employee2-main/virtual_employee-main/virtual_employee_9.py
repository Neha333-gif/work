
# 9. Virtual Employee Agent


import os
import json
import random
import sys
import io
from datetime import datetime
from crewai import Crew, Task, Agent, LLM, Process
from crewai.tools import BaseTool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Type
import uvicorn

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Index
from sqlalchemy.orm import declarative_base, sessionmaker



# Fix Windows encoding issues for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

my_llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

memory = {"research": {}, "business_topic": {}, "resources": []}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])

class ResourceSchema(BaseModel):
    resource: str = Field(..., description="The resource to manage: 'meeting', 'email', or 'reports'")
    details: str = Field("No details provided", description="Additional details like subject, recipient, or notes")
    priority: str = Field("Normal", description="Priority level: 'High', 'Normal', or 'Low'")


class manage_resources(BaseTool):
    name: str = "manage_resources"
    description: str = "manage the resources"
    args_schema: Type[BaseModel] = ResourceSchema

    def _run(self, resource: str, details: str = "No details", priority: str = "Normal") -> str:
        if "resources" not in memory: 
            memory["resources"] = []
        res_lower = resource.lower()
        if any(keyword in res_lower for keyword in ["meeting", "email", "reports"]):
            target = "meeting" if "meeting" in res_lower else "email" if "email" in res_lower else "reports"
            memory["resources"].append({
                "resource": target,
                "time": datetime.now().isoformat(),
                "action": "scheduled " + target,
                "details": details,
                "priority": priority
            })
            return f"Resource {target} ({priority} priority) scheduled: {details}"
        else:
            return "invalid resource entered. Please use 'meeting', 'email', or 'reports'"

class TopicSchema(BaseModel):
    topic: str = Field(..., description="The topic to research")


class conduct_research(BaseTool):
    name: str = "conduct_research"
    description: str = "conduct research based on the given topic"
    args_schema: Type[BaseModel] = TopicSchema

    def _run(self, topic: str) -> str:
        memory["research"]["scheduled"] = topic.lower()
        memory["research"]["status"] = "ongoing"
        memory["research"]["time"] = datetime.now().isoformat()
        return f"Research on {topic} scheduled at {datetime.now().isoformat()}"

class ResultSchema(BaseModel):
    results: str = Field(..., description="The results to store")

class store_results(BaseTool):
    name: str = "store_results"
    description: str = "store the results of the research"
    args_schema: Type[BaseModel] = ResultSchema

    def _run(self, results: str) -> str:
        memory["research"]["results"] = results
        memory["research"]["status"] = "completed"
        memory["research"]["time"] = datetime.now().isoformat()
        return f"Research results stored at {datetime.now().isoformat()}"

class InsightSchema(BaseModel):
    business_topic_and_insights: str = Field(..., description="The topic and insights in format 'topic: insights'")

class provide_insights(BaseTool):
    name: str = "provide_insights"
    description: str = "provide insights based on the research"
    args_schema: Type[BaseModel] = InsightSchema

    def _run(self, business_topic_and_insights: str) -> str:
        if ":" in business_topic_and_insights:
            business_topic, insights = business_topic_and_insights.split(":", 1)
        else:
            business_topic, insights = business_topic_and_insights, "No insights provided"
        
        engine = create_engine("sqlite:///memory.db", connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)
        Base = declarative_base()


        class Insight(Base):
            __tablename__ = "insights"
            id = Column(Integer, primary_key=True)
            topic = Column(String)
            content = Column(Text)

        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        db.add(Insight(topic=business_topic.strip(), content=insights.strip()))
        db.commit()
        db.close()

        if "business_topic" not in memory: 
            memory["business_topic"] = {}
        memory["business_topic"][business_topic.strip()] = {
            "insights": insights.strip(),
            "status": "completed",
            "time": datetime.now().isoformat()
        }
        return f"insights on {business_topic.strip()} provided at {datetime.now().isoformat()}"


manage_resources = manage_resources()
conduct_research = conduct_research()
store_results = store_results()
provide_insights = provide_insights()


    # -------agents --------

store_resource_agent = Agent(
    role = "store resource agent",
    goal = "manages the resources",
    backstory = "expert in managing resources",
    llm = my_llm,
    tools = [manage_resources],
    allow_delegation = False,
    verbose = True)

research_agent = Agent(
    role = "research agent",
    goal = "conducts research and provide detailed results in bullet points",
    backstory = "expert in conducting research and providing detailed results in bullet points",
    llm = my_llm,
    tools = [conduct_research, store_results],
    allow_delegation = False,
    verbose = True)

insight_agent = Agent(
    role = "insight agent",
    goal = "provides insights based on the research",
    backstory = "expert in providing insights based on the research",
    llm = my_llm,
    tools = [provide_insights],
    allow_delegation = False,
    verbose = True)


    # ----tasks----

task_1 = Task(
        description = "manage the resources",
        expected_output = "resources managed successfully",
        agent = store_resource_agent)

task_2 = Task(
        description = "conduct research on the given topic",
        expected_output = "research conducted successfully",
        agent = research_agent)

task_3 = Task(
        description = "provide insights based on the research",
        expected_output = "insights provided successfully",
        agent = insight_agent)

crew = Crew(
    agents = [store_resource_agent, research_agent, insight_agent],
    tasks = [task_1, task_2, task_3],
    verbose = True)

@app.get("/insights")
def get_insights():
    try:
        engine = create_engine("sqlite:///memory.db", connect_args={"check_same_thread": False})
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT id, topic, content FROM insights ORDER BY id DESC"))
            rows = [{"id": row[0], "topic": row[1], "content": row[2]} for row in result]
        return {"insights": rows, "total": len(rows)}
    except Exception as e:
        return {"insights": [], "total": 0, "error": str(e)}

@app.get("/run")
def run_crew():
    try:
        result = crew.kickoff()
        return {"status": "success", "result": str(result), "memory": memory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory")
def get_memory():
    return memory

@app.get("/config")
def get_config():
    return {
        "agents": [
            {"role": a.role, "goal": a.goal, "backstory": a.backstory}
            for a in [store_resource_agent, research_agent, insight_agent]
        ],
        "tasks": [
            {"description": t.description, "expected_output": t.expected_output, "agent": t.agent.role}
            for t in [task_1, task_2, task_3]
        ],
        "tools": [
            {"name": "manage_resources", "description": "Manage meetings, emails, and reports"},
            {"name": "conduct_research", "description": "Perform deep research on any topic"},
            {"name": "store_results", "description": "Save research data to knowledge base"},
            {"name": "provide_insights", "description": "Extract business value from research"}
        ]
    }

@app.post("/execute_task")
def execute_specific_task(topic: str):
    # Dynamically update task descriptions if needed
    task_2.description = f"conduct research on the topic: {topic}"
    try:
        result = crew.kickoff()
        return {"status": "success", "result": str(result), "memory": memory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

    

