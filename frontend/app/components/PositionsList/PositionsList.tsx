import styles from "./PositionsList.module.css";

type Position = {
  symbol: string;
  qty: string;
  current_price: string;
  avg_entry_price: string;
  market_value: string;
  unrealized_pl: string;
}; 

type Props = {
  positions: Position[];
  onClosePosition: (symbol: string) => void;
};

export default function PositionsList({positions, onClosePosition,}: Props) {
  return (
    <div className={styles.positionsPanel}>
      <h3>Open Positions</h3>

      <div className={`${styles.positionRow} ${styles.positionRowHeader}`}>
          <div>Symbol</div>
          <div>Qty</div>
          <div>Market Value</div>
          <div>P/L</div>
          <div />
      </div>
      <div className={styles.positionsList}>
        {positions.map((position) => (
            <div
                key={position.symbol}
                className={styles.positionRow}
            >
                <div>{position.symbol}</div>

                <div>{position.qty}</div>

                <div>
                ${Number(position.market_value).toFixed(2)}
                </div>

                <div
                className={
                    Number(position.unrealized_pl) >= 0
                    ? "profit"
                    : "loss"
                }
                >
                ${Number(position.unrealized_pl).toFixed(2)}
                </div>

                <button
                  className={styles.closeButton}
                  onClick={async () => {
                    const confirmed = window.confirm(
                      `Are you sure you want to close your entire ${position.symbol} position?\n\nThis will sell all shares currently held.`
                    );

                    if (confirmed) {
                      await onClosePosition(position.symbol);
                    }
                  }}
                >
                  ✕
                </button>
            </div>
          ))} 
      </div>
    </div>
  );
}