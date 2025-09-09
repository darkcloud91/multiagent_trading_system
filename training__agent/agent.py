# agents/training_agent.py

from google.adk.agents import LlmAgent
from tools.training_tools import train_model_handler_multi, check_existing_model

training_agent = LlmAgent(
    model="gemini-1.5-flash",
    name="training_agent",
    description="An agent responsible for managing the machine learning model training pipeline.",
    instruction="""
    You are an agent that manages the machine learning model training pipeline.

    When the user wants to train a model, your first step is ALWAYS to use the
    `check_existing_model` tool to see if a model already exists.

    - If a model **already exists**, inform the user and ask for explicit confirmation
      before proceeding. You must warn them that retraining is a long, computationally
      intensive process that will **overwrite the existing model**.
    
    - If no model exists, you can proceed more directly.

    Your main training tool is 'train_model_handler_multi'. You can ask the user if they
    want to train on a specific list of stocks; otherwise, it will default to the NASDAQ-100.
    
    After any tool finishes, report the final success or failure message back to the user.
    """,
    tools=[
        check_existing_model,
        train_model_handler_multi,
    ]
)