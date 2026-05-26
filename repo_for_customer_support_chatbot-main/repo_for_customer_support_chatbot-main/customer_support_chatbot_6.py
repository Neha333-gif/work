# 6. Customer Support Resolution Agent

# More advanced than chatbots:

# Understands context across conversations
# Resolves issues
# Escalates intelligently
# ➡️ Real-world enterprise use case


import os
import json
import random
from datetime import datetime
from crewai import Crew, Task, Agent, LLM, Process
from crewai.tools import BaseTool

my_llm = LLM(model="gemini/gemini-flash-latest", api_key="AIzaSyAYfXkDm3036-HZpl676_WPAY_6XE73fw4")

memory = {"conversation_memory": [], "escalated_tickets": [], "ticket": {}}

past_users = {}
fraud_msg = []

class understand_context(BaseTool):
    name : str = "understand_context"
    description : str = " understands the context of the given conversation "

    def _run(self, message : str, name : str, user_id : int) -> str:
        memory["conversation_memory"].append({
            "user_name" : name,
            "user_id" : user_id,
            "message" : message,
            "time" : datetime.now().isoformat(),
            "width" : len(message.split())})
        past_users[name] = message
        return json.dumps(memory["conversation_memory"])

class issue_resolver(BaseTool):
    name : str = "issue resolver"
    description : str = "resolves the issues and creates a ticket and stores it"

    def _run(self, issue: str) -> str:
        print(f"The issue of {issue} has been resolved")
        # create the ticket for the issue resolve and store in memory
        memory["ticket"] = {
            "id" : f"ticket{random.randint(100, 300)}",
            "issue" : issue,
            "resolved_timeings" : datetime.now().isoformat(),
            "word count" : len(issue.split())}
        return json.dumps(memory["ticket"])

class escalation_human(BaseTool):
    name : str = "human escalator"
    description : str = "escalates teh issue to humans when feedback is required"

    def _run(self, issue: str) -> str:
        print("the issue {issue} is being escalated to human feedback")
        memory["escalated_tickets"].append({
        "Ticket_id" : memory["ticket"]["id"],
        "issue" : issue,
        "time_of_escalation" : datetime.now().isoformat(),
        "resolution_status" : "escalated to human for feedback"})
        return json.dumps(memory["escalated_tickets"])

immeadiate_responses = []
mediocre_responses = []
technical_response = []

class msg_group(BaseTool):
    name : str = "message polarity"
    description : str = "detect the polarity of the given message"
    
    def _run(self, message: str) -> str:
        words = message.split()
        for word in words:
            if word in ["urgent", "quick"]:
                immeadiate_responses.extend(word)
            elif word in ["inquiry", "help"]:
                mediocre_responses.extend(word)
            elif word in ["bug", "fix", "technical"]:
                technical_response.extend(word)
            else:
                pass
        return "message polarity processed"


class compare_users(BaseTool):
    name : str = "compare_users_tools"
    description : str = "compare users based on profile and mark it if they are same"

    def _run(self, name: str, id: str, message: str) -> str:
        if id in past_users:
            past_users[id].append(message)
        else:
            past_users[id] = [message]
        return json.dumps(past_users)

polarity_msg = {}
sensitive_info = []
normal_inbox = []

class recieve_feedback(BaseTool):
    name : str = "feedback_tool"
    description : str = "gives feedback based on response"

    def _run(self, feedback: str = "") -> str:
        feeds = feedback.split()
        for feed in feeds:
            if feed in ["bad", "not good"]:
                polarity_msg[feed] = "negative, need to be improved"
            else:
                polarity_msg[feed] = "positive"
        return json.dumps(polarity_msg)

class security_detection(BaseTool):
    name : str = "security_detection_tool"
    description : str = "scans the message for fraud or sensitive data"

    def _run(self, msg: str) -> str:
        tokens = msg.lower()
        
        is_fraud = any(x in tokens for x in ["offer", "win", "cash", "prize", "lottery", "fraud", "free"])
        is_sensitive = any(x in tokens for x in ["bank", "account", "number", "pan", "card", "aadar", "ssn", "password", "pin"])
        
        if is_fraud: fraud_msg.append(msg)
        if is_sensitive: sensitive_info.append(msg)
        if not is_fraud and not is_sensitive: normal_inbox.append(msg)
        
        return json.dumps({"fraud": is_fraud, "sensitive": is_sensitive})


# agents 

context_agent = Agent(
    role = "context understanding",
    goal = "to understand the mesaage contexts",
    backstory = " you rae an expert in understanding the contexts for 10 years",
    tools = [understand_context()],
    llm = my_llm,
    allow_delegation = False,
    verbose = True)
    

issue_agent = Agent(
    role = "issue resolving specialist",
    goal = "to resolve the issues",
    backstory = " you are an expert in resolving issues of customers for 10 years",
    tools = [issue_resolver()],
    llm = my_llm,
    allow_delegation = False,
    verbose = True)

human_agent = Agent(
    role = "human escalation",
    goal = "to escalate the issues to humans when feedback is required",
    backstory = "you are an expert in escalating the issues to humans when feedback is required",
    tools = [escalation_human()],
    llm = my_llm,
    allow_delegation = False,
    verbose = True)

insight_agent = Agent(
    role = "insight_agent",
    goal = "create the insights based on the messages recieved",
    backstory = "you are expert in creating insights based on messages from last 20 years which in turn helps to understabd user issues",
    tools = [compare_users()],
    llm = my_llm,
    allow_delegation = False,
    verbose = True)

master_support_agent = Agent(
    role="Customer Support Specialist",
    goal="Handle context, security, and resolution in a single pass",
    backstory="Expert at multi-tasking while staying within strict security guidelines.",
    tools=[understand_context(), security_detection(), issue_resolver(), escalation_human()],
    llm=my_llm,
    allow_delegation=False,
    verbose=True)

Task1 = Task(
    description = "understand the context of the given conversation and store it",
    expected_output = "A list of conversations context",
    agent = context_agent)

Task2 = Task(
    description = "resolve the issues and create a ticket and stores it",
    expected_output = "A list of tickets",
    agent = issue_agent,
    context = [Task1])

Task3 = Task(
    description = "escalate the issues to humans when feedback is required",
    expected_output = "A list of escalated tickets",
    agent = human_agent,
    context = [Task2])

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="."), name="static")

class SupportRequest(BaseModel):
    message: str
    user_name: str = "Anonymous"
    user_id: str = "0"

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/support")
def run_support(req: SupportRequest):
    # Single comprehensive task to minimize API calls
    TaskFullProcess = Task(
        description=f"""
        Process this support request: '{req.message}' for user '{req.user_name}' (ID: {req.user_id}).
        
        Follow these steps exactly:
        1. Call 'understand_context' with the message, user_name, and user_id.
        2. Call 'security_detection_tool' with the message to check for sensitive/fraud data.
        3. Call 'issue resolver' with a summary of the issue to create a ticket.
        4. If 'security_detection_tool' found sensitive/fraud data OR if the user is angry, call 'human escalator'.
        """,
        expected_output="A concise summary of all actions taken and the final resolution.",
        agent=master_support_agent)

    try:
        crew = Crew(
            agents=[master_support_agent],
            tasks=[TaskFullProcess],
            verbose=True)

        result = crew.kickoff()
        return {"status": "success", "result": str(result), **get_state()}
    except Exception as e:
        return {"status": "error", "result": str(e), **get_state()}

@app.get("/state")
def get_state():
    return {
        "conversation_memory": memory["conversation_memory"],
        "escalated_tickets": memory["escalated_tickets"],
        "ticket": memory["ticket"],
        "fraud_msg": fraud_msg,
        "sensitive_info": sensitive_info,
        "normal_inbox": normal_inbox,
        "past_users": past_users,
        "immeadiate_responses": immeadiate_responses,
        "mediocre_responses": mediocre_responses,
        "technical_response": technical_response,
        "polarity_msg": polarity_msg,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=44081)
