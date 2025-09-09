from google.adk.tools.tool_context import ToolContext
from typing import Dict, Any, List



def calculate_performance(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Calcula y devuelve un informe de rendimiento completo para todas las posiciones cerradas.
    
    Returns:
        Un diccionario con métricas de rendimiento financiero y del modelo.
    """
    closed_positions: List[Dict] = tool_context.state.get("closed_positions", [])

    if not closed_positions:
        return {"status": "success", "message": "No hay operaciones cerradas para analizar."}

    # === Métricas de Rendimiento Financiero ===
    total_pnl = sum(pos.get('profit_loss', 0) for pos in closed_positions)
    total_gains = sum(pos.get('profit_loss', 0) for pos in closed_positions if pos.get('profit_loss', 0) > 0)
    total_losses = abs(sum(pos.get('profit_loss', 0) for pos in closed_positions if pos.get('profit_loss', 0) < 0))
    wins = sum(1 for pos in closed_positions if pos.get('profit_loss', 0) > 0)
    total_operations = len(closed_positions)
    win_rate = wins / total_operations if total_operations else 0.0
    profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

    # === Métricas de Rendimiento del Modelo ===
    correct_buy_predictions = 0
    correct_sell_predictions = 0
    total_buy_predictions = 0
    total_sell_predictions = 0

    for pos in closed_positions:
        entry_price = pos.get('entry_price')
        exit_price = pos.get('exit_price')
        predicted_buy_price = pos.get('predicted_buy_price')
        predicted_sell_price = pos.get('predicted_sell_price')

        if predicted_buy_price:
            total_buy_predictions += 1
            # Predicción de compra correcta si el precio real de entrada fue menor que el predicho.
            if entry_price and entry_price <= predicted_buy_price:
                correct_buy_predictions += 1

        if predicted_sell_price:
            total_sell_predictions += 1
            # Predicción de venta correcta si el precio real de salida fue mayor que el predicho.
            if exit_price and exit_price >= predicted_sell_price:
                correct_sell_predictions += 1
                
    buy_accuracy = correct_buy_predictions / total_buy_predictions if total_buy_predictions else 0.0
    sell_accuracy = correct_sell_predictions / total_sell_predictions if total_sell_predictions else 0.0

    # === Construcción del Informe Final ===
    report = {
        "status": "success",
        "message": "Informe de rendimiento generado.",
        "financial_metrics": {
            "total_operations": total_operations,
            "total_pnl": round(total_pnl, 2),
            "total_gains": round(total_gains, 2),
            "total_losses": round(total_losses, 2),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2)
        },
        "model_performance_metrics": {
            "buy_prediction_accuracy": round(buy_accuracy, 2),
            "sell_prediction_accuracy": round(sell_accuracy, 2)
        },
        "closed_positions_details": closed_positions
    }
    
    return report