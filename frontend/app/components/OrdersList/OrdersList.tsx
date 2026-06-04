"use client";

import { useEffect, useState } from "react";
import styles from "./OrdersList.module.css";

type Order = {
  id?: string;
  symbol?: string;
  side?: string;
  qty?: number;
  status?: string;
};

export default function OrdersList() {
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchOrders = async () => {
    try {
      const res = await fetch("http://localhost:8000/orders");
      const data = await res.json();

      setOrders(data.orders || []);
    } catch (err) {
      console.error("Failed to fetch orders:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOrders();

    const interval = setInterval(fetchOrders, 5000);

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div>Loading orders...</div>;
  }

  return (
    <div className={styles.container}>
      <h3 className={styles.title}>Orders:</h3>

      {orders.map((order, index) => (
        <div
          key={order.id || index}
          className={styles.order}
        >
          <div>{order.symbol}</div>
          <div>{order.side}</div>
          <div>{order.qty}</div>
          <div>{order.status}</div>
        </div>
      ))}
    </div>
  );
}