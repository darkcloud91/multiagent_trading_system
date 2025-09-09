from google.adk.agents import LlmAgent
from tools.inference_tools import infer_multi


inference_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="inference_agent",
    description="An agent that uses a trained Transformer model to predict stock market movements.",
    instruction="""
    You are an AI-powered trading inference agent. Your primary function is to
    predict future price movements for stocks.

    When the user asks for predictions, you must use the 'infer_multi' tool.
    This tool can handle requests for specific stocks or for the entire NASDAQ-100
    if no specific symbols are mentioned.
    
    After the tool runs, it will return a ranked list of the most promising stocks
    and the path to a JSON file with the full results. Your job is to present a
    clear and concise summary of this output to the user, highlighting the top-ranked
    stocks and mentioning that the full details have been saved.
    """,
    tools=[infer_multi]
)
