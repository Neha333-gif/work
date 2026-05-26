# 5. Autonomous Content Creation Pipeline

# Agent workflow:

# Keyword research → blog writing → SEO optimization → publishing
# ➡️ Shows end-to-end execution without human intervention


import os
import json
import random
from datetime import datetime
from crewai import Crew, Task, Agent, LLM, Process
from crewai.tools import BaseTool


my_llm = LLM(model="gemini/gemini-2.0-flash", api_key="AIzaSyAYfXkDm3036-HZpl676_WPAY_6XE73fw4")


agent_handoff = {}
memory = []

tracked_agents = {
            "keyword_research" : {"status" : None, "starting_time" : None, "end_time" : None, "time_taken" : None},
            "save_blog" : {"status" : None, "starting_time" : None, "end_time" : None, "time_taken" : None},
            "seo_optimizer" : {"status" : None, "starting_time" : None, "end_time" : None, "time_taken" : None},
            "publish_content" : { "status" : None, "starting_time" : None, "end_time" : None, "time_taken" : None}}

from pydantic import BaseModel, Field
from typing import Type

class KeywordResearchInput(BaseModel):
    topic: str = Field(..., description="The topic to research keywords for.")

class keyword_research(BaseTool):
    name : str = "keyword_research_tool"
    description : str = "research on different keywords available"
    args_schema: Type[BaseModel] = KeywordResearchInput

    def _run(self, topic: str) -> str:
        keyword = [
            {"keyword" : f"best {topic}", "volume" : random.randint(100, 500), 
            "difficulty" : random.randint(10, 50), "cpc" : random.randint(20, 50)},
            {"keyword" : f"tutorial on {topic}", "volume" : random.randint(200, 500), 
            "difficulty" : random.randint(10, 80), "cpc" : random.randint(10, 50)}]
        return json.dumps({"topic": topic, "keywords": keyword}, indent=2)



class SaveBlogInput(BaseModel):
    content: str = Field(..., description="The content of the blog post to save.")

class save_blog(BaseTool):
    name : str = "save blog tool"
    description : str = "save the written blog post"
    args_schema: Type[BaseModel] = SaveBlogInput

    def _run(self, content: str) -> str:
        blog = {"id" : f"blog_{random.randint(1000,9999)}", "content" : content,
                "word_count" : len(content.split()), "created_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        return json.dumps(blog, indent=2)


class SEOOptimizerInput(BaseModel):
    content: str = Field(..., description="The content to optimize for SEO.")

class seo_optimizer(BaseTool):
    name : str = "seo optimizer tool"
    description : str = "optimize content for SEO and return score"
    args_schema: Type[BaseModel] = SEOOptimizerInput

    def _run(self, content: str) -> str:
        report = {"seo_score" : random.randint(75, 95), "readability" : random.randint(60, 90),
                  "keyword_density" : round(random.uniform(1.5, 3.5), 2),
                  "meta_title" : content[:60], "meta_description" : content[:155]}
        return json.dumps(report, indent=2)


class PublishContentInput(BaseModel):
    content: str = Field(..., description="The content to publish.")

class publish_content(BaseTool):
    name : str = "publish content tool"
    description : str = "publish the finalized blog post"
    args_schema: Type[BaseModel] = PublishContentInput

    def _run(self, content: str) -> str:
        record = {"post_id" : f"pub_{random.randint(10000,99999)}",
                  "url" : f"https://myblog.com/posts/{random.randint(100,999)}",
                  "published_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  "status" : "published"}
        return json.dumps(record, indent=2)


keyword_research_tool = keyword_research()


class AgentTrackerInput(BaseModel):
    agent_name: str = Field(..., description="The name of the agent to track.")
    status: str = Field(..., description="The status of the agent (start, running, completed, failed).")

class agent_tracker_tool(BaseTool):
    name : str = "agent_tracker_tool"
    description : str = "It tracks the behavior of each agent and generates report"
    args_schema: Type[BaseModel] = AgentTrackerInput
    
    def _run(self, agent_name: str, status: str):
        if agent_name in tracked_agents:
            if status == "start":
                tracked_agents[agent_name]["status"] = "started_running"
                tracked_agents[agent_name]["starting_time"] = datetime.now()
            elif status == "running":
                tracked_agents[agent_name]["status"] = "running"
            elif status == "completed":
                tracked_agents[agent_name]["status"] = "completed"
                tracked_agents[agent_name]["end_time"] = datetime.now()
                if tracked_agents[agent_name]["starting_time"]:
                    duration = (tracked_agents[agent_name]["end_time"] - tracked_agents[agent_name]["starting_time"]).total_seconds()
                    tracked_agents[agent_name]["time_taken"] = f"{duration} seconds"
            elif status == "failed":
                tracked_agents[agent_name]["status"] = "failed"
                tracked_agents[agent_name]["end_time"] = datetime.now()
                if tracked_agents[agent_name]["starting_time"]:
                    duration = (tracked_agents[agent_name]["end_time"] - tracked_agents[agent_name]["starting_time"]).total_seconds()
                    tracked_agents[agent_name]["time_taken"] = f"{duration} seconds"
        
        return json.dumps(tracked_agents, indent=2, default=str)

class ManageHandoffsInput(BaseModel):
    agent_name: str = Field(..., description="The name of the agent.")
    status: str = Field(..., description="The status of the handoff.")

class manage_handoffs(BaseTool):
    name : str = "handoff_managing_tool"
    description : str = "basically handles the handoff of task and completion of tasks"
    args_schema: Type[BaseModel] = ManageHandoffsInput

    def _run(self, status: str, agent_name: str):
        if agent_name == "keyword_research" and status == "start":
            memory.append(f"The keyword_research agent has started at {datetime.now()}!")
        elif agent_name == "save_blog" and status == "start":
            memory.append(f"Keyword search agent ended!")
            memory.append(f"The process has been handed off to save_blog at {datetime.now()}")
        elif agent_name == "seo_optimizer" and status == "start":
            memory.append(f"Save_blog agent ended!")
            memory.append(f"The process has been handed off to seo_optimizer at {datetime.now()}")
        elif agent_name == "publish_content" and status == "start":
            memory.append(f"Content is being published at {datetime.now()}!")
        
        return f"Handoff logged for {agent_name} with status {status}"

           

# agents

keyword_agent = Agent(
    role = "keyword research specialist",
    goal = "find the best keywords for a given topic",
    backstory = "expert in SEO keyword research with 10 years of experience",
    llm = my_llm,
    tools = [keyword_research(), agent_tracker_tool(), manage_handoffs()],
    allow_delegation = False, verbose = True)

writer_agent = Agent(
    role = "blog writer",
    goal = "write an engaging blog post based on keyword research",
    backstory = "expert content writer specializing in tech and AI topics",
    llm = my_llm,
    tools = [save_blog(), agent_tracker_tool(), manage_handoffs()],
    allow_delegation = False, verbose = True)

seo_agent = Agent(
    role = "SEO specialist",
    goal = "optimize blog content for maximum search engine visibility",
    backstory = "expert in on-page SEO, meta tags, and keyword density optimization",
    llm = my_llm,
    tools = [seo_optimizer(), agent_tracker_tool(), manage_handoffs()],
    allow_delegation = False, verbose = True)

publisher_agent = Agent(
    role = "content publisher",
    goal = "publish finalized SEO optimized blog posts to the platform",
    backstory = "expert in content publishing and social media promotion",
    llm = my_llm,
    tools = [publish_content(), agent_tracker_tool(), manage_handoffs()],
    allow_delegation = False, verbose = True)


# tasks

Task_1 = Task(
    description = "do keyword research for the topic: Generative AI for Businesses",
    expected_output = "a list of keywords with volume, difficulty and cpc",
    agent = keyword_agent)

Task_2 = Task(
    description = "write a blog post based on the keyword research from Task 1. save it using the save blog tool.",
    expected_output = "a complete blog post saved with word count and id",
    agent = writer_agent,
    context = [Task_1])

Task_3 = Task(
    description = "optimize the blog post from Task 2 for SEO using the seo optimizer tool",
    expected_output = "an SEO report with score, readability, keyword density and meta tags",
    agent = seo_agent,
    context = [Task_2])

Task_4 = Task(
    description = "publish the SEO optimized blog post from Task 3 using the publish content tool",
    expected_output = "a publishing confirmation with post url and status",
    agent = publisher_agent,
    context = [Task_3])


def log_step(step_output):
    memory.append(f"Progress: {str(step_output)[:200]}...")

crew = Crew(
    agents = [keyword_agent, writer_agent, seo_agent, publisher_agent],
    tasks = [Task_1, Task_2, Task_3, Task_4],
    verbose = True,
    step_callback=log_step)

if __name__ == "__main__":
    result = crew.kickoff()
    print(f"after crew kickoff : {result}")
