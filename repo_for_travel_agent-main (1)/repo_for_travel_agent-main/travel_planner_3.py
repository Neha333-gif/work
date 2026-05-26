import warnings
import os
from dotenv import load_dotenv
load_dotenv()
from crewai import Crew, Task, Agent, LLM
from crewai.tools import BaseTool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List, Optional
import json

# Suppress warnings
warnings.filterwarnings('ignore')

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
my_llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# Global state (for demo purposes)
history = {}
budget = 20000

class PreferenceRequest(BaseModel):
    name: str
    travel_type: str
    food_type: str
    comfort: str
    destination: str

class choose_type(BaseTool):
    name: str = "choose_type_tool"
    description: str = "tool to choose and validate types of travel, food, comfort etc"
    
    def _run(self, name: str, travel_type: str, food_type: str, comfort: str):
        history["name"] = name
        history["travel_type"] = travel_type
        history["food_type"] = food_type
        history["comfort"] = comfort
        return f"Preferences recorded for {name}."

class budget_tracker(BaseTool):
    name: str = "budget_tracker_tool"
    description: str = "tool to track budget and provide feedback"

    def _run(self, expenses: Dict[str, float]):
        return "Budget analysis complete. Within limits."

class weather(BaseTool):
    name: str = "weather_tool"
    description: str = "tool to get weather information and adjust itinerary"

    def _run(self, location: str):
        return f"Weather in {location} is perfect for travel."

# Initialize tools
choose_type_tool = choose_type()
budget_tracker_tool = budget_tracker()
weather_tool = weather()

@app.post("/plan")
async def plan_trip(req: PreferenceRequest):
    try:
        # Update global history
        history.update(req.dict())

        # Define Agents
        search_flights = Agent(
            role="Flight Specialist",
            goal=f"Find flights to {req.destination}",
            backstory="Expert at finding deals.",
            llm=my_llm,
            tools=[budget_tracker_tool],
            verbose=True)

        book_hotels = Agent(
            role="Hotel Specialist",
            goal=f"Find hotels in {req.destination}",
            backstory="Expert at luxury stays.",
            llm=my_llm,
            tools=[budget_tracker_tool],
            verbose=True)

        optimize_itinerary = Agent(
            role="Travel Planner",
            goal=f"Create a 3-day itinerary for {req.destination}",
            backstory="Master of itineraries.",
            llm=my_llm,
            tools=[weather_tool],
            verbose=True)

        # Define Tasks
        task1 = Task(description=f"Flights to {req.destination}", expected_output="Flights list", agent=search_flights)
        task2 = Task(description=f"Hotels in {req.destination}", expected_output="Hotel list", agent=book_hotels)
        task3 = Task(description=f"3-day itinerary for {req.destination}", expected_output="Itinerary", agent=optimize_itinerary, context=[task1, task2])

        # Run Crew
        crew = Crew(
            agents=[search_flights, book_hotels, optimize_itinerary],
            tasks=[task1, task2, task3],
            verbose=True)

        result = crew.kickoff()
        return {
            "status": "success",
            "result": str(result),
            "history": history
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
