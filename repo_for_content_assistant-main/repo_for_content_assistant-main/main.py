from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from datetime import datetime
import threading
from content_assistant_5 import (
    crew, tracked_agents, memory, 
    keyword_agent, writer_agent, seo_agent, publisher_agent,
    Task_1, Task_2, Task_3, Task_4
)

app = FastAPI(title="Content Creation AI Pipeline")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to keep track of the latest result
execution_state = {
    "is_running": False,
    "result": None,
    "error": None,
    "start_time": None,
    "end_time": None
}

def run_crew_task(topic: str):
    global execution_state
    execution_state["is_running"] = True
    execution_state["start_time"] = datetime.now().isoformat()
    execution_state["result"] = None
    execution_state["error"] = None
    
    # Update task description with the topic if provided
    if topic:
        Task_1.description = f"do keyword research for the topic: {topic}"
    
    try:
        result = crew.kickoff()
        execution_state["result"] = str(result)
    except Exception as e:
        execution_state["error"] = str(e)
    finally:
        execution_state["is_running"] = False
        execution_state["end_time"] = datetime.now().isoformat()

@app.get("/")
async def root():
    return {"message": "Content Assistant API is running"}

@app.post("/start")
async def start_pipeline(background_tasks: BackgroundTasks, topic: str = "Generative AI for Businesses"):
    if execution_state["is_running"]:
        return {"status": "error", "message": "Pipeline is already running"}
    
    background_tasks.add_task(run_crew_task, topic)
    return {"status": "started", "topic": topic}

@app.get("/status")
async def get_status():
    return {
        "execution": execution_state,
        "agents": tracked_agents,
        "logs": memory
    }

@app.get("/reset")
async def reset_state():
    global execution_state
    execution_state = {
        "is_running": False,
        "result": None,
        "error": None,
        "start_time": None,
        "end_time": None
    }
    # Reset tracked agents
    for agent in tracked_agents:
        tracked_agents[agent] = {"status": None, "starting_time": None, "end_time": None, "time_taken": None}
    # Reset memory
    memory.clear()
    return {"status": "reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
