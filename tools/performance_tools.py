# tools/performance_tools.py
from google.adk.tools.tool_context import ToolContext
from typing import Dict, Any, List
import numpy as np

def calculate_performance(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Calculates and returns a comprehensive performance report for all closed positions.

    This tool analyzes the list of closed positions stored in the agent's state,
    computing both financial profitability metrics and prediction model performance metrics.
    It expects each closed position dictionary to contain the 'prediction_class' and
    'prediction_probability' that led to the trade.
    
    Returns:
        A dictionary containing financial and model performance metrics.
    """
    closed_positions: List[Dict] = tool_context.state.get("closed_positions", [])

    if not closed_positions:
        return {"status": "success", "message": "No closed positions to analyze."}

    # === Financial Performance Metrics ===
    total_pnl = sum(pos.get('profit_loss', 0) for pos in closed_positions)
    total_gains = sum(pos.get('profit_loss', 0) for pos in closed_positions if pos.get('profit_loss', 0) > 0)
    total_losses = abs(sum(pos.get('profit_loss', 0) for pos in closed_positions if pos.get('profit_loss', 0) < 0))
    wins = sum(1 for pos in closed_positions if pos.get('profit_loss', 0) > 0)
    total_operations = len(closed_positions)
    win_rate = wins / total_operations if total_operations else 0.0
    profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

    # === Prediction Performance Metrics ===
    
    # 1. Prediction Hit Rate
    hits = 0
    for pos in closed_positions:
        # Assumes trades are opened based on "UP" predictions
        if pos.get('prediction_class') == 'UP' and pos.get('profit_loss', 0) > 0:
            hits += 1
    
    prediction_hit_rate = hits / total_operations if total_operations else 0.0

    # 2. Confidence / P&L Correlation
    correlation = 0.0
    if total_operations > 1:
        probabilities = [pos.get('prediction_probability', 0.0) for pos in closed_positions]
        pnls = [pos.get('profit_loss', 0.0) for pos in closed_positions]
        if np.std(probabilities) > 0 and np.std(pnls) > 0:
            correlation = np.corrcoef(probabilities, pnls)[0, 1]

    # 3. P&L Analysis by Confidence Buckets
    buckets = {"60-70%": [], "70-80%": [], "80-90%": [], "90-100%": []}
    for pos in closed_positions:
        prob = pos.get('prediction_probability', 0.0) * 100
        if 60 <= prob < 70:
            buckets["60-70%"].append(pos.get('profit_loss', 0))
        elif 70 <= prob < 80:
            buckets["70-80%"].append(pos.get('profit_loss', 0))
        elif 80 <= prob < 90:
            buckets["80-90%"].append(pos.get('profit_loss', 0))
        elif 90 <= prob <= 100:
            buckets["90-100%"].append(pos.get('profit_loss', 0))

    avg_pnl_by_bucket = {
        bucket: round(sum(pnls) / len(pnls), 2) if pnls else 0.0
        for bucket, pnls in buckets.items()
    }
    
    # === Final Report Build ===
    report = {
        "status": "success",
        "message": "Performance report generated.",
        "financial_metrics": {
            "total_operations": total_operations,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2)
        },
        "prediction_performance_metrics": {
            "prediction_hit_rate": f"{round(prediction_hit_rate * 100, 2)}%",
            "confidence_pnl_correlation": round(correlation, 3),
            "avg_pnl_by_confidence": avg_pnl_by_bucket
        },
        "closed_positions_details": closed_positions
    }
    
    return report