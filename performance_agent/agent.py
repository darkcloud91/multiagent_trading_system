
from google.adk.agents import LlmAgent
from tools.performance_tools import calculate_performance


performance_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="performance_agent",
    description="An agent specialized in analyzing the financial performance of trading operations.",
    instruction="""
    You are a specialized agent for measuring trading performance.
    Your main task is to calculate and report key metrics for closed trading positions.
    
    Use the 'calculate_performance_tool' to generate a detailed report that includes financial metrics
    (like total PnL and win rate) and model-specific metrics (like prediction accuracy).
    
    Your responses should be clear, informative, and focused on providing a comprehensive analysis of the trades.
    After completing a task print the result to the console and delegate back to the trading_manager_agent so it can await the next request by the user.
    """,
    tools=[calculate_performance]
)
