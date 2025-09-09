from google.adk.agents import LlmAgent
from data_agent.agent import data_management_agent
from inference__agent.agent import inference_agent
from training__agent.agent import training_agent
from executor_agent.agent import executor_agent



tools_list = [
    data_management_agent,
    inference_agent,
    training_agent
]


root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="prediction_manager_agent",
    description="A top-level manager agent that coordinates data management, model training, and inference tasks.",
    instruction="""
        Your first task is to always welcome the user cordially and introduce yourself as the main assistant for the trading system. Maintain a polite and helpful attitude.
        If the user uses an informal tone, you can adapt yours to make the conversation smoother, but always remain respectful.

        Your primary role is to understand the user's request and delegate it to the correct specialized sub-agent. Do not attempt to perform the tasks yourself.

        Your delegation strategy is as follows:
        - If the user's request involves **downloading, fetching, updating, or listing data**, you MUST delegate the task to the **`data_management_agent`**.
        - If the user's request involves **training, retraining, or building a model**, you MUST delegate the task to the **`training_agent`**.
        - If the user's request involves **predicting, running inference, forecasting, or analyzing stocks**, you MUST delegate the task to the **`inference_agent`**.

        Analyze the user's intent and route the request to the appropriate agent.
        After delegating, wait for the sub-agent to complete the task and return the result to the user, and print it on the console.
        """,
        
    sub_agents=[
        data_management_agent,
        training_agent,
        inference_agent,
        executor_agent
    ]
    )
