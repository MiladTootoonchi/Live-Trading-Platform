"use client";

import { useEffect, useState } from "react";
import styles from "./OrderingPanel.module.css";

const VALID_ORDER_TYPES = [
    "market",
    "limit",
    "stop",
    "stop_limit",
    "trailing_stop",
];

export default function OrderingPanel() {
    const [symbol, setSymbol] = useState("");
    const [qty, setQty] = useState("");
    const [side, setSide] = useState("buy");
    const [orderType, setOrderType] = useState("market");

    const [validation, setValidation] = useState({
        is_valid: false,
        message: "",
    });

    useEffect(() => {
        if (!symbol || !qty) {
            setValidation({
                is_valid: false,
                message: "",
            });

            return;
        }

        const timeout = setTimeout(async () => {
            try {
                const params = new URLSearchParams({
                    symbol,
                    qty,
                    order_type: orderType,
                });

                const response = await fetch(
                    `http://localhost:8000/validate_order?${params}`
                );

                const data = await response.json();

                setValidation(data);
            } catch {
                setValidation({
                    is_valid: false,
                    message: "Failed to validate order",
                });
            }
        }, 500);

        return () => clearTimeout(timeout);
    }, [symbol, qty, orderType]);

    const placeOrder = async () => {
        try {
            const response = await fetch(
                "http://localhost:8000/place_order",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        symbol,
                        qty: Number(qty),
                        side,
                        order_type: orderType,
                    }),
                }
            );

            const data = await response.json();

            alert(`Order submitted: ${data.order_id}`);
        } catch {
            alert("Failed to place order");
        }
    };

    return (
        <div className={styles.orderingPanel}>
            <h3 className={styles.h3}>Order Here</h3>

            <input
                type="text"
                placeholder="Symbol"
                value={symbol}
                onChange={(e) =>
                    setSymbol(e.target.value.toUpperCase())
                }
            />

            <select
                className={styles.select}
                value={side}
                onChange={(e) => setSide(e.target.value)}
            >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
            </select>

            <input
                type="number"
                min="1"
                placeholder="Quantity"
                value={qty}
                onChange={(e) => setQty(e.target.value)}
            />

            <select
                className={styles.select}
                value={orderType}
                onChange={(e) => setOrderType(e.target.value)}
            >
                {VALID_ORDER_TYPES.map((type) => (
                    <option
                        key={type}
                        value={type}
                    >
                        {type}
                    </option>
                ))}
            </select>

            {validation.message && (
                <p
                    className={
                        validation.is_valid
                            ? "profit"
                            : "loss"
                    }
                >
                    {validation.message}
                </p>
            )}

            <button
                className={`${styles.button} ${
                    validation.is_valid ? styles.validButton : styles.disabledButton
                }`}
                onClick={placeOrder}
                disabled={!validation.is_valid}
            >
                Submit Order
            </button>
        </div>
    );
}