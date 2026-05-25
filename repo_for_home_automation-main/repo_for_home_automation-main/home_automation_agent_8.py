from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
import random
from datetime import datetime
from collections import defaultdict
import json 
from typing import List, Dict, Any

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

my_llm = LLM(model="groq/mixtral-8x7b-32768", api_key="YOUR_API_KEY_HERE")

# Global memory to persist state
memory = {"electric_components": {}, "commands": ""}

class reciever_agent(BaseTool):
    name: str = "natural language processing agent"
    description: str = "receives natural language commands and stores it"

    def _run(self, command: str) -> str:
        memory["commands"] = command 
        return command

class DeviceControlTool(BaseTool): 
    name: str = "device_power_tool"
    description: str = "Turn devices ON or OFF. Use this ONLY for basic power switching. For speed, temperature, or levels, use the 'user_preference_tool'."

    def _run(self, command: str) -> str: 
        print(f"DEBUG: DeviceControlTool called with command: {command}")
        command_lower = command.lower()
        if not isinstance(memory["electric_components"], dict):
            memory["electric_components"] = {}
            
        for word in command_lower.split():
            if word == "lights":
                if "on" in command_lower: memory["electric_components"]["lights"] = True 
                elif "off" in command_lower: memory["electric_components"]["lights"] = False 
            elif word == "fan":
                if "on" in command_lower: memory["electric_components"]["fan"] = True 
                elif "off" in command_lower: memory["electric_components"]["fan"] = False 
            elif word == "ac":
                if "on" in command_lower: memory["electric_components"]["ac"] = True 
                elif "off" in command_lower: memory["electric_components"]["ac"] = False 
            elif word == "geyser":
                if "on" in command_lower: memory["electric_components"]["geyser"] = True 
                elif "off" in command_lower: memory["electric_components"]["geyser"] = False 
            elif word == "heater":
                if "on" in command_lower: memory["electric_components"]["heater"] = True 
                elif "off" in command_lower: memory["electric_components"]["heater"] = False 
            elif word == "robot":
                if "clean" in command_lower or "resume" in command_lower: memory["electric_components"]["robot"] = True 
                elif "stop" in command_lower or "pause" in command_lower: memory["electric_components"]["robot"] = False 
            else: 
                return f"Invalid command"
        
        return json.dumps(memory["electric_components"])


class UserPreferenceTool(BaseTool):
    name: str = "user_preference_tool"
    description: str = "Adjust SPEED, TEMPERATURE, or LEVEL for devices (fan, ac, geyser, heater). Use this for 'low', 'medium', 'high' settings."

    def _run(self, command: str) -> str:
        print(f"DEBUG: UserPreferenceTool called with command: {command}")
        command_lower = command.lower()
        if not isinstance(memory["electric_components"], dict):
            memory["electric_components"] = {}
            
        for device in ["fan", "ac", "geyser", "heater"]:
            if device in command_lower:
                for pref in ["low", "medium", "high"]:
                    if pref in command_lower:
                        # Automatically turn on the device if a preference is set
                        memory["electric_components"][device] = {"status": True, "user_preference": pref}
            
        return json.dumps(memory["electric_components"])



# Instantiate tools
reciever_tool = reciever_agent()
home_automation_tool = DeviceControlTool()
user_preference_tool = UserPreferenceTool()



# Agent definition
home_automation_agent_inst = Agent(
    role = "Home System Controller",
    goal = "Accurately control home devices based on user commands by using the provided tools.",
    backstory = "You are the central processor for Aura Home OS. You must ALWAYS use the 'device_power_tool' for simple on/off, and the 'user_preference_tool' for any speed, level, or intensity adjustments (low/medium/high). If a user asks to set a speed, it implies the device should be turned on as well.",
    llm = my_llm,
    tools = [reciever_tool, home_automation_tool, user_preference_tool],
    allow_delegation = False, 
    verbose = True)


# FastAPI Endpoints
class CommandRequest(BaseModel):
    command: str

@app.get("/state")
async def get_state():
    return memory["electric_components"]

@app.post("/command")
async def run_command(request: CommandRequest):
    # Task to process the command
    task = Task(
        description = f"Process this user command: {request.command}",
        expected_output = "The result of the home automation action",
        agent = home_automation_agent_inst)

    crew = Crew(
        agents = [home_automation_agent_inst],
        tasks = [task],
        verbose = True)

    try:
        result = crew.kickoff()
        return {"result": str(result), "state": memory["electric_components"]}
    except Exception as e:
        error_msg = str(e)
        if "503" in error_msg or "high demand" in error_msg.lower():
            return {"result": "The AI service is currently overloaded. Please try again in a few moments.", "state": memory["electric_components"]}
        return {"result": f"An error occurred: {error_msg}", "state": memory["electric_components"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
