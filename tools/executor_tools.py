from datetime import datetime, timezone
from google.adk.tools.tool_context import ToolContext
from typing import Dict, Any, List
from trading_utils.alpaca_adapter import AlpacaAPI 
from alpaca.trading.models import Order, Position
from alpaca.data.models import Quote
from alpaca.trading.enums import OrderStatus


async def get_open_positions(tool_context: ToolContext) -> Dict[str, Any]:
        """
        This tool gets the user's open positions.
        
        Returns:
        A dictionary with the all the open positions.
        """
        alpaca_api = AlpacaAPI()
        try:
            open_positions_raw = alpaca_api.get_open_positions()
            if not isinstance(open_positions_raw, list) or not all(isinstance(pos, Position) for pos in open_positions_raw):
                tool_context.state["open_positions"] = []
                return {"status": "error", "message": "No se pudieron obtener las posiciones abiertas correctamente."}
            open_positions: List[Position] = open_positions_raw

            if not open_positions:
                tool_context.state["open_positions"] = []
                return {"status": "success", "message": "No tienes posiciones abiertas."}

            # Convertir los objetos Position a diccionarios serializables
            positions_data = []
            for pos in open_positions:
                positions_data.append({
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "current_price": float(pos.current_price)
                })

            tool_context.state["open_positions"] = positions_data
            
            return {
                "status": "success",
                "message": f"Se han obtenido {len(positions_data)} posiciones abiertas.",
                "open_positions": positions_data
            }
        except Exception as e:
            return {"status": "error", "message": f"Error al obtener las posiciones abiertas: {e}"}
        
def sell_order(
    symbol: str, 
    qty: float,
    entry_price: float,
    predicted_buy_price: float,
    predicted_sell_price: float,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Submit a stock sell order and record the closed position,
    using predicted prices for analysis.

    Args:
        symbol: The stock ticker, e.g., "AAPL", "GOOG".
        qty: The quantity of shares to sell, e.g., 1.0 or 25.5.
        entry_price: The actual price at which the position was opened.
        predicted_buy_price: The price predicted by the model at the time of purchase.
        predicted_sell_price: The price predicted by the model at the time of sale.
        tool_context: The tool context object, passed automatically by the agent.
    
    Returns:
        A dictionary with the submission result and the performance data.
    """
    alpaca_api = AlpacaAPI()
    try:
        api_response: Order = alpaca_api.submit_order(symbol=symbol, quantity=qty, side="sell")
        # Si la respuesta de la API es None (la llamada falló), devuelve un error.
        if api_response is None:
            return {
                "status": "error",
                "message": "La llamada a la API de Alpaca falló. Por favor, revisa el log de la consola para más detalles."
            }
        # Check if the order was filled to calculate PnL and save the closed position
        if api_response.status in [OrderStatus.FILLED, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW]:
            if api_response.filled_avg_price is None:
                market_data: Quote  = alpaca_api.get_market_data(symbol)
                estimated_price = None
                if market_data:
                # Usar el precio de demanda para las compras
                    estimated_price = market_data.ask_price
                    exit_price = float(api_response.filled_avg_price) if api_response.filled_avg_price else estimated_price
                    profit_loss = (exit_price - entry_price) * qty
            #profit_loss = exit_price 
            closed_position = {
                "symbol": symbol,
                "status": str(api_response.status.value),
                "order_id": str(api_response.id),
                "client_order_id": api_response.client_order_id,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "profit_loss": profit_loss,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "predicted_buy_price": predicted_buy_price, # Store the prediction
                "predicted_sell_price": predicted_sell_price, # Store the prediction
            }

            closed_positions_list = tool_context.state.get("closed_positions", [])
            closed_positions_list.append(closed_position)
            tool_context.state["closed_positions"] = closed_positions_list

            message = f"La orden de venta para {qty} acciones de {symbol} ha sido completada. La posición fue cerrada con una ganancia/pérdida de {profit_loss:.2f}."
        
        else:
            message = f"La orden de venta para {qty} acciones de {symbol} ha sido enviada con el estado: {api_response.status.value}. No se ha cerrado la posición."

        # Construct the response dictionary
        response_dict = {
            "status": str(api_response.status.value),
            "order_id": str(api_response.id),
            "client_order_id": api_response.client_order_id,
            "message": message,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "profit_loss": profit_loss,
            "exit_date": datetime.now(timezone.utc).isoformat(),
            "filled_avg_price": float(api_response.filled_avg_price) if api_response.filled_avg_price else None,
            "filled_qty": float(api_response.filled_qty) if api_response.filled_qty else None,
            "predicted_buy_price": predicted_buy_price,
            "predicted_sell_price": predicted_sell_price,
        }
        return response_dict

    except Exception as e:
        return {
            "status": "error",
            "message": f"Hubo un problema al enviar la orden de venta: {e}"
        }


def buy_order(
    symbol: str, 
    qty: float,
    predicted_buy_price: float,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Submit a stock buy order to the Alpaca API.

    Args:
        symbol: The stock ticker, e.g., "AAPL", "GOOG".
        qty: The quantity of shares to trade, e.g., 1.0 or 25.5.
        predicted_buy_price: The predicted buy price for the stock.
        context: The tool context object, passed automatically by the agent.
    
    Returns:
        A dictionary with the submission result.
    """
    alpaca_api = AlpacaAPI()
    try:
        
        api_response: Order  = alpaca_api.submit_order(symbol=symbol, quantity=qty, side="buy")
        
        if api_response is None:
            return {
                "status": "error",
                "message": "La llamada a la API de Alpaca falló. Por favor, revisa el log de la consola para más detalles."
        
             }
        
        # Lógica para guardar la orden en el estado si ha sido procesada
        if api_response.status in [OrderStatus.FILLED, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW]:
            if api_response.filled_avg_price is None:
                market_data: Quote  = alpaca_api.get_market_data(symbol)
                estimated_price = None
                if market_data:
                # Usar el precio de demanda para las compras
                    estimated_price = market_data.ask_price
            
            filled_price = float(api_response.filled_avg_price) if api_response.filled_avg_price else estimated_price
            
            new_order = {
                "status": str(api_response.status.value),
                "order_id": str(api_response.id),
                "message": f"La orden de compra para {qty} acciones de {symbol} ha sido enviada con el estado: {api_response.status.value}.",
                "client_order_id": api_response.client_order_id,
                "created_at": api_response.created_at.isoformat() if api_response.created_at else None,
                "filled_avg_price": filled_price,
                "filled_qty": float(api_response.filled_qty) if api_response.filled_qty else None,
                "filled_at": api_response.filled_at.isoformat() if api_response.filled_at else None,
                "symbol": symbol,
                "qty": qty,
                "predicted_buy_price": predicted_buy_price, # <-- Campo añadido
            }
            orders_list = tool_context.state.get("orders", [])
            orders_list.append(new_order)
            tool_context.state["orders"] = orders_list

        # Construye el diccionario de respuesta para el agente
        response_dict = {
            "status": str(api_response.status.value),
            "order_id": str(api_response.id),
            "message": f"La orden de compra para {qty} acciones de {symbol} ha sido enviada con el estado: {api_response.status.value}.",
            "client_order_id": api_response.client_order_id,
            "created_at": api_response.created_at.isoformat() if api_response.created_at else None,
            "filled_avg_price": float(api_response.filled_avg_price) if api_response.filled_avg_price else estimated_price,
            "filled_qty": float(api_response.filled_qty) if api_response.filled_qty else None,
            "filled_at": api_response.filled_at.isoformat() if api_response.filled_at else None,
            "symbol": symbol,
            "qty": qty,
            "predicted_buy_price": predicted_buy_price,
        }
        return response_dict
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Hubo un problema al enviar la orden de compra: {e}"
        }
        
        
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

def cancel_orders(order_ids: List[str]) -> Dict[str, Any]:
    """
    Cancels one or more pending trading orders.

    Args:
        order_ids: A list of order IDs to be canceled. For a single order,
                   it should be a list with one element (e.g., ["some-id"]).
    """
    # Ya no necesitamos comprobar el tipo, asumimos que siempre es una lista.
    ids_to_cancel = order_ids

    alpaca_api = AlpacaAPI()
    successful_cancellations = []
    failed_cancellations = []

    for order_id in ids_to_cancel:
        try:
            cancellation_response = alpaca_api.cancel_order(order_id)
            if cancellation_response and cancellation_response["status"] == "success":
                successful_cancellations.append(order_id)
            else:
                failed_cancellations.append(order_id)
        except Exception:
            failed_cancellations.append(order_id)

    # El resto de la lógica de resumen se mantiene igual
    total_requested = len(ids_to_cancel)
    total_success = len(successful_cancellations)
    total_failed = len(failed_cancellations)
    
    message = f"Resumen de la cancelación: {total_success} de {total_requested} órdenes canceladas con éxito."
    if total_failed > 0:
        message += f" {total_failed} fallaron."

    return {
        "status": "partial_success" if total_failed > 0 and total_success > 0 else "success" if total_failed == 0 else "error",
        "message": message,
        "successful_ids": successful_cancellations,
        "failed_ids": failed_cancellations
    }

def list_pending_orders() -> Dict[str, Any]:
    """
    Use this tool to get a list of all trading orders that are currently open
    (e.g., 'accepted', 'new') and have not yet been filled. These are the orders
    that can be canceled.
    """
    alpaca_api = AlpacaAPI()
    try:
        pending_orders: List[Order] = alpaca_api.get_pending_orders()

        if not pending_orders:
            return {"status": "success", "message": "No hay ninguna orden pendiente para cancelar."}

        orders_data = []
        for order in pending_orders:
            orders_data.append({
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "status": order.status.value,
                "submitted_at": order.submitted_at.isoformat()
            })
        
        return {
            "status": "success",
            "message": f"Se han encontrado {len(orders_data)} órdenes pendientes.",
            "pending_orders": orders_data
        }
    except Exception as e:
        return {"status": "error", "message": f"Error al obtener las órdenes pendientes: {e}"}







