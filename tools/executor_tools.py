# tools/executor_tools.py

import os
import csv
from datetime import datetime, timezone
from google.adk.tools.tool_context import ToolContext
from typing import Dict, Any, List, Optional
from trading_utils.alpaca_adapter import AlpacaAPI 
from alpaca.trading.models import Order, Position
from alpaca.data.models import Quote
from alpaca.trading.enums import OrderStatus


async def get_open_positions(tool_context: ToolContext) -> Dict[str, Any]:
    """
    This tool gets the user's open positions from their portfolio.
    
    Returns:
        A dictionary with all the open positions. The 'open_positions' key contains
        a list where each position includes the symbol, quantity, current price, cost basis,
        unrealized profit/loss, and the total market_value of the holding.
    """
    alpaca_api = AlpacaAPI()
    try:
        open_positions_raw = alpaca_api.get_open_positions()
        if not isinstance(open_positions_raw, list) or not all(isinstance(pos, Position) for pos in open_positions_raw):
            tool_context.state["open_positions"] = []
            return {"status": "error", "message": "Could not retrieve open positions correctly."}
        open_positions: List[Position] = open_positions_raw

        if not open_positions:
            tool_context.state["open_positions"] = []
            return {"status": "success", "message": "You have no open positions."}

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
            "message": f"Retrieved {len(positions_data)} open positions.",
            "open_positions": positions_data
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting open positions: {e}"}

def sell_order(
    symbol: str, 
    entry_price: float,
    tool_context: ToolContext,
    qty: Optional[float] = None,
    notional: Optional[float] = None
) -> Dict[str, Any]:
    """
    Submit a stock sell order by specifying either a quantity of shares or a notional (dollar) amount.

    Args:
        symbol: The stock ticker, e.g., "AAPL", "GOOG".
        qty: The quantity of shares to sell. Use this OR notional.
        notional: The dollar amount to sell. Use this OR qty.
        entry_price: The actual price at which the position was opened.
        tool_context: The tool context object, passed automatically by the agent.
    """
    alpaca_api = AlpacaAPI()
    try:
        api_response: Order = alpaca_api.submit_order(symbol=symbol, side="sell", qty=qty, notional=notional)
        if api_response is None:
            return {"status": "error", "message": "Alpaca API call failed. Please check the console log for details."}
        
        exit_price = 0.0
        profit_loss = 0.0
        
        if api_response.status in [OrderStatus.FILLED, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW]:
            if api_response.filled_avg_price is not None:
                exit_price = float(api_response.filled_avg_price)
            else:
                market_data: Quote = alpaca_api.get_market_data(symbol)
                if market_data:
                    exit_price = market_data.bid_price
            
            if exit_price > 0:
                order_qty = float(api_response.filled_qty) if api_response.filled_qty else qty
                if order_qty:
                    profit_loss = (exit_price - entry_price) * order_qty

            orders_list = tool_context.state.get("orders", [])
            if not orders_list:
                print("Context 'orders' is empty. Attempting to load from data/orders.csv")
                CSV_PATH = os.path.join("data", "orders.csv")
                if os.path.exists(CSV_PATH):
                    try:
                        with open(CSV_PATH, 'r', newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            loaded_orders = []
                            for row in reader:
                                for key in ['qty', 'notional', 'filled_avg_price', 'prediction_probability']:
                                    if row.get(key) is not None and row[key] != '':
                                        try:
                                            row[key] = float(row[key])
                                        except (ValueError, TypeError):
                                            pass
                                loaded_orders.append(row)
                        
                        orders_list = loaded_orders
                        # ✅ **CORRECCIÓN: Guardar la lista cargada de nuevo en el contexto**
                        tool_context.state["orders"] = orders_list
                        print(f"Successfully loaded {len(orders_list)} orders from CSV into context.")
                    except Exception as e:
                        print(f"Error reading orders.csv: {e}")

            original_prediction = {"prediction_class": "UNKNOWN", "prediction_probability": 0.0}
            for order in reversed(orders_list):
                if order.get("symbol") == symbol and order.get("status") in ["filled", "accepted"] and order.get("side") == "buy":
                    original_prediction["prediction_class"] = order.get("prediction_class", "UNKNOWN")
                    original_prediction["prediction_probability"] = order.get("prediction_probability", 0.0)
                    break

            closed_position = {
                "symbol": symbol, "status": str(api_response.status.value), "order_id": str(api_response.id),
                "client_order_id": api_response.client_order_id, "qty": qty or notional, "entry_price": entry_price,
                "exit_price": exit_price, "profit_loss": profit_loss, "exit_date": datetime.now(timezone.utc).isoformat(),
                "prediction_class": original_prediction["prediction_class"],
                "prediction_probability": original_prediction["prediction_probability"],
            }

            closed_positions_list = tool_context.state.get("closed_positions", [])
            closed_positions_list.append(closed_position)
            tool_context.state["closed_positions"] = closed_positions_list
            message = f"Sell order for {symbol} has been completed. Position closed with a P/L of {profit_loss:.2f}."
        else:
            message = f"Sell order for {symbol} has been submitted with status: {api_response.status.value}."

        return {
            "status": str(api_response.status.value), "order_id": str(api_response.id),
            "client_order_id": api_response.client_order_id, "message": message,
        }

    except Exception as e:
        return {"status": "error", "message": f"An error occurred while submitting the sell order: {e}"}

def buy_order(
    symbol: str,
    prediction_class: str,
    prediction_probability: float,
    tool_context: ToolContext,
    qty: Optional[float] = None,
    notional: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Submit a stock buy order by specifying either a quantity of shares or a notional (dollar) amount.

    Args:
        symbol: The stock ticker, e.g., "AAPL", "GOOG".
        prediction_class: The prediction class from the model (e.g., "UP").
        prediction_probability: The confidence of the prediction (e.g., 0.85).
        tool_context: The tool context object, passed automatically by the agent.
        qty: The quantity of shares to buy. Use this OR notional.
        notional: The dollar amount to invest. Use this OR qty.
    """
    alpaca_api = AlpacaAPI()
    try:
        api_response: Order = alpaca_api.submit_order(symbol=symbol, side="buy", qty=qty, notional=notional)
        if api_response is None:
            return {"status": "error", "message": "Alpaca API call failed."}
        
        estimated_price = None
        if api_response.filled_avg_price is None:
            market_data: Quote = alpaca_api.get_market_data(symbol)
            if market_data:
                estimated_price = market_data.ask_price
        
        filled_price = float(api_response.filled_avg_price) if api_response.filled_avg_price else estimated_price
        
        new_order = {
            "status": str(api_response.status.value), "order_id": str(api_response.id),
            "symbol": symbol, "side": "buy", "qty": qty, "notional": notional, 
            "filled_avg_price": filled_price,
            "prediction_class": prediction_class,
            "prediction_probability": prediction_probability,
        }
        
        orders_list = tool_context.state.get("orders", [])
        orders_list.append(new_order)
        tool_context.state["orders"] = orders_list

        DATA_DIR = "data"
        CSV_PATH = os.path.join(DATA_DIR, "orders.csv")
        os.makedirs(DATA_DIR, exist_ok=True)
        
        write_header = not os.path.exists(CSV_PATH)
        
        try:
            with open(CSV_PATH, 'a', newline='') as csvfile:
                fieldnames = new_order.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if write_header:
                    writer.writeheader()
                writer.writerow(new_order)
            print(f"Order {api_response.id} successfully saved to {CSV_PATH}")
        except IOError as e:
            print(f"Error saving order to CSV: {e}")

        return {
            "status": str(api_response.status.value), "order_id": str(api_response.id),
            "message": f"Buy order for {qty or notional} of {symbol} has been submitted with status: {api_response.status.value}.",
        }
        
    except Exception as e:
        return {"status": "error", "message": f"An error occurred while submitting the buy order: {e}"}

def cancel_orders(order_ids: List[str]) -> Dict[str, Any]:
    """
    Cancels one or more pending trading orders.

    Args:
        order_ids: A list of order IDs to be canceled. For a single order,
                   it should be a list with one element (e.g., ["some-id"]).
    """
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

    total_requested = len(ids_to_cancel)
    total_success = len(successful_cancellations)
    total_failed = len(failed_cancellations)
    
    message = f"Cancellation summary: {total_success} of {total_requested} orders canceled successfully."
    if total_failed > 0:
        message += f" {total_failed} failed."

    return {
        "status": "partial_success" if total_failed > 0 and total_success > 0 else "success" if total_failed == 0 else "error",
        "message": message,
        "successful_ids": successful_cancellations,
        "failed_ids": failed_cancellations
    }

# tools/executor_tools.py

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
            return {"status": "success", "message": "There are no pending orders to cancel."}

        orders_data = []
        for order in pending_orders:
            # ✅ --- INICIO DE LA CORRECCIÓN ---
            # Comprobar si qty o notional son None antes de convertirlos
            qty_value = float(order.qty) if order.qty is not None else None
            notional_value = float(order.notional) if order.notional is not None else None
            side_value = order.side.value if order.side is not None else "unknown"

            orders_data.append({
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": qty_value,
                "notional": notional_value, # Añadido para más claridad
                "side": side_value,
                "status": order.status.value,
                "submitted_at": order.submitted_at.isoformat()
            })
            # ✅ --- FIN DE LA CORRECCIÓN ---
        
        return {
            "status": "success",
            "message": f"Found {len(orders_data)} pending orders.",
            "pending_orders": orders_data
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting pending orders: {e}"}
    
async def get_available_cash() -> Dict[str, Any]:
    """
    Obtiene el saldo de efectivo disponible para operar en la cuenta de Alpaca.
    
    Returns:
        Un diccionario con el efectivo disponible y el estatus.
    """
    alpaca_api = AlpacaAPI()
    try:
        # ✅ Llama al nuevo método del AlpacaAPI para obtener el saldo
        cash = alpaca_api.get_available_cash()
        
        if cash is not None:
            return {
                "status": "success",
                "message": f"Tu saldo de efectivo disponible para operar es: {cash:.2f} USD.",
                "available_cash": cash
            }
        
        return {"status": "error", "message": "No se pudo recuperar la información del saldo en efectivo de la cuenta."}
    
    except Exception as e:
        return {"status": "error", "message": f"Error al obtener el saldo en efectivo: {e}"}             