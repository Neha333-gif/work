from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

import sys
import os
# Add your Groq API key here or set it as an environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your_api_key_here")





GLOBAL_INCOME = 50000
GLOBAL_MAX_EXPENSE = 500
import random
from datetime import datetime
from collections import defaultdict
import json 


my_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY"),
    max_tokens=2048
    )









# memory 
memory = {"transactions" : [], "budgets" : [], "alerts" : [], "insights" : []}
# since agents only understand natural language 

thought = []
insights = []
history_budget = []


class FinanaceAgentTool(BaseTool):
    name : str = "fetch_transactions"
    description : str = "Fetches finance information of latest expenses from banking transaction api. Provide 'run' as argument."

    def _run(self, command: str) -> str:    
        categories = ["groceries", "dining out", "transportation", "utilities", "entertainment", "shopping", "health", "others"]
        transactions = [{"id" : random.randint(100, 999), "category" : random.choice(categories),
        "price" : random.randint(100, GLOBAL_MAX_EXPENSE), "date" : datetime.now().strftime("%Y-%m-%d")} for _ in range(5)]
        memory["transactions"].append(transactions)
        
        thought.append(f"the transactions are as follows {transactions}")
        return json.dumps(transactions)

class BudgetAgentTool(BaseTool):
    name : str = "track_budget_health"
    description : str = "Provide 'run' as argument to analyze the current budget and alert unusual spending."

    def _run(self, command: str) -> str:
        if not memory["transactions"]:
            return "No transactions found."
            
        latest_transactions = memory["transactions"][-1]
        total_spent = sum(t["price"] for t in latest_transactions)
        by_category = defaultdict(float)
        for t in latest_transactions:
            by_category[t["category"]] += t["price"]

        thought.append(f"total_spend amount is {total_spent}")
        
        alerts = []
        if total_spent > 1000:
            msg = f"alert : spending too much ({total_spent}). Need to reduce spending."
            insights.append(msg)
            alerts.append(msg)
            memory["alerts"].append(msg)
        else:
            msg = f"spendings are under control! ({total_spent})"
            insights.append(msg)
            alerts.append(msg)

        return f"Total spent: {total_spent}. Category breakdown: {dict(by_category)}. Alerts: {alerts}"

class SaveTool(BaseTool):
    name : str = "save_insight"
    description : str = "Save the insights and reasoning. Provide the insight text as the argument."

    def _run(self, insights_str: str) -> str:
        memory["insights"].append({"insights" : insights_str, "date_time" : datetime.now().isoformat()})
        return "Insight saved successfully."

class ReadTool(BaseTool):
    name : str = "read_memory"
    description : str = "Read the historical data and long-term memory for strong portfolio impact. Pass 'all' as argument."

    def _run(self, command: str) -> str:
        res = []
        for k, v in memory.items():
            res.append(f"{k}: {len(v)} items")
        return "Memory contents summary: " + ", ".join(res) + f" | Recent insights: {memory['insights'][-3:] if memory['insights'] else 'None'}"

class caluculate_monthly_expenses(BaseTool):
    name : str = "monthly_expense_caluculator"
    description : str = "caluculate monthly expenses"

    def _run(self, command: str = "run") -> str:
        if not memory["transactions"]:
            return "No transaction duration mentioned"
        caluculate_expenses = {}
        for batch in memory["transactions"]:
            for t in batch:
                date = t["date"][:7]  # yyyy-mm
                if date not in caluculate_expenses:
                    caluculate_expenses[date] = 0
                caluculate_expenses[date] += t["price"]
        return json.dumps(caluculate_expenses)


class highest_expense_tool(BaseTool):
    name : str = "highest_expense_finder"
    description : str = "it caluculates the product with highest expense in that month"

    def _run(self, command: str = "run") -> str:
        if not memory["transactions"]:
            return "No transaction data available"
        highest_expense : int = 0
        for batch in memory["transactions"]:
            for t in batch:
                if t["price"] > highest_expense:
                    highest_expense = t["price"]
        return highest_expense



class budget_suggest_tool(BaseTool):
    name : str = "budget_suggester"
    description : str = "suggest budgets based on income and spending"

    def _run(self, income: str = "50000") -> str:
        total_spent = sum(t["price"] for batch in memory["transactions"] for t in batch)
        income = float(GLOBAL_INCOME)
        if total_spent > income:
            memory["alerts"].append(f"you spend more than your income")
        elif total_spent < income and total_spent > (80/100)*income:
            memory["alerts"].append(f"you are nearing the 80%+ spending of income")
        else:
            memory["alerts"].append(f"you are well within your income")
        return memory["alerts"][-1]


class alert_handling_agent(BaseTool):
    name : str = "alert_handler"
    description : str = "handle the alerts and unexpected expenses"

    def _run(self, command: str = "run") -> str:
        alerts_tokens = ["warning", "immediate", "limit exceeded"]
        if not memory["alerts"]:
            return "No alerts found."
        combined = " ".join(str(a) for a in memory["alerts"]).lower()
        if any(word in combined for word in alerts_tokens):
            percent_reduce = random.randint(10, 30)
            return f"reduce spending by {percent_reduce}% to stay on track"
        return f"Alerts: {memory['alerts'][-3:]}" 

class habit_detection(BaseTool):
    name : str = "habit_detection_tool"
    description : str = "detect the most bought items"

    def _run(self, command: str = "run") -> str:
        if not memory["transactions"]:
            return "No transactions data available"
        item_list = {}
        for batch in memory["transactions"]:
            for item in batch:
                if item["category"] not in item_list:
                    item_list[item["category"]] = 0
                item_list[item["category"]] += 1
        max_bought_item = max(item_list, key=lambda x: item_list[x])
        return f"you bought {max_bought_item} more than once"

class behavior_detection(BaseTool):
    name : str = "behavior_detection_tool"
    description : str = "detect the user behavior"

    def _run(self, command: str = "run") -> str:
        if not memory["transactions"]:
            return "No transaction data available"
        weekend_spending = 0
        for batch in memory["transactions"]:
            for t in batch:
                day = datetime.strptime(t["date"], "%Y-%m-%d").strftime("%A")
                if day.lower() == "sunday":
                    weekend_spending += t["price"]
        weekend_spending_msg = f"you spend on {weekend_spending} on weekends"
        memory["insights"].append(weekend_spending_msg)

        cat_count = {}
        for batch in memory["transactions"]:
            for t in batch:
                cat_count[t["category"]] = cat_count.get(t["category"], 0) + 1
        max_category = max(cat_count, key = lambda x: cat_count[x])
        most_bought_category_msg = f"most bought category is {max_category}"
        memory["insights"].append(most_bought_category_msg)
        return f"{weekend_spending_msg} | {most_bought_category_msg}"


def save_memory():
    with open("memory.json", "w") as f: 
        json.dump(memory, f)

def load_memory():
    global memory
    try:
        with open("memory.json", "r") as f:
            memory = json.load(f)
    except:
        pass

# tools
financetool = FinanaceAgentTool()
Budgettool = BudgetAgentTool()
savetool = SaveTool()
readtool = ReadTool()
caluculatetool = caluculate_monthly_expenses()
highest_expense_tool = highest_expense_tool()
alert_handling_tool = alert_handling_agent()
habit_detection_tool = habit_detection()
budget_suggest_tool = budget_suggest_tool()
behavior_detection_tool = behavior_detection()

# agents — each agent gets only the tools it strictly needs
finance_agent = Agent(
    role = "personal finance agent",
    goal = "Fetch transactions and save a concise summary of the spending.",
    backstory = "Expert CFP who tracks spending using live transaction data.",
    tools = [financetool, savetool],
    verbose = True,
    llm = my_llm,
    allow_delegation = False)

budget_agent = Agent(
    role = "personal budget agent",
    goal = "Suggest a budget plan based on current spending and income.",
    backstory = "Expert CFP who optimizes personal budgets and investments.",
    tools = [budget_suggest_tool, savetool],
    verbose = True,
    llm = my_llm,
    allow_delegation = False)

habit_detection_agent = Agent(
    role = "personal habit detection agent",
    goal = "Detect the most frequently bought category.",
    backstory = "Behavioral finance expert who spots spending patterns.",
    tools = [habit_detection_tool],
    verbose = True,
    llm = my_llm,
    allow_delegation = False)

expense_alert_agent = Agent(
    role = "personal expense alert agent",
    goal = "Analyze spending health and trigger alerts for overspending.",
    backstory = "Risk-aware CFP who monitors budgets and raises alerts.",
    tools = [Budgettool, alert_handling_tool],
    verbose = True,
    llm = my_llm,
    allow_delegation = False)



    
# tasks 
Task_1 = Task(
    description = "Fetch the latest transactions and analyze the categories. Note any unusual spending. Save your insights.",
    expected_output = "3 short bullet points on current transactions and spending",
    agent = finance_agent) 

Task_2 = Task(
    description = "Read historical memory and the latest budget analysis. Suggest a budget and investment plan.",
    expected_output = "5 concise bullet points on budget and investment suggestions.",
    agent = budget_agent) 


Task_3 = Task(
    description = "Handle the alerts and unexpected expenses. Be concise.",
    expected_output = "3 bullet points on alerts and recommended actions.",
    agent = expense_alert_agent)

Task_4 = Task(
    description = "detect the most bought items",
    expected_output = "most bought item and the count of it",
    agent = habit_detection_agent, )

Task_5 = Task(
    description = "Save a brief summary of all findings to memory.",
    expected_output = "Confirmed: summary saved to memory.",
    agent = finance_agent,
    dependencies = [Task_1, Task_2, Task_3, Task_4])

crew = Crew(
    agents = [finance_agent, budget_agent, expense_alert_agent, habit_detection_agent],
    tasks = [Task_1, Task_2, Task_3, Task_4, Task_5],
    verbose = True,
    planning = False,
    max_rpm = 2)



def run_tracker(income_val=50000, max_expense_val=500):
    global GLOBAL_INCOME, GLOBAL_MAX_EXPENSE
    GLOBAL_INCOME = income_val
    GLOBAL_MAX_EXPENSE = max_expense_val
    
    global memory, thought, insights, history_budget
    memory["transactions"].clear()
    memory["budgets"].clear()
    memory["alerts"].clear()
    memory["insights"].clear()
    thought.clear()
    insights.clear()
    history_budget.clear()
    
    results = crew.kickoff()
    return {
        "results": str(results),
        "memory": {
            "alerts": memory["alerts"],
            "insights": memory["insights"],
            "transactions": memory["transactions"],
            "budgets": memory["budgets"]
        },
        "thought": thought
    }

if __name__ == "__main__":
    res = run_tracker()
    print("results are : ", res["results"])
