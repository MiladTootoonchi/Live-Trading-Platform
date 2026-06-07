import styles from "./AccountMetricsPanel.module.css";

type AccountMetrics = {
  equity: number;
  cash: number;
  unrealized_pnl: number;
  realized_pnl: number;
  buying_power: number;
  maintenance_margin: number;
  initial_margin: number;
  account_leverage: number;
  margin_cushion: number;
};

type Props = {
  metrics: AccountMetrics;
};

const currency = (value: number) =>
  value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });

export default function AccountMetricsPanel({
  metrics,
}: Props) {
  return (
    <div className={styles.panel}>
      <div className={styles.section}>
        <h3>Account & Performance</h3>

        <div className={styles.highlightRow}>
          <span className={styles.label}>Equity</span>
          <span className={styles.highlightValue}>
            {currency(metrics.equity)}
          </span>
        </div>

        <div className={styles.row}>
          <span className={styles.label}>Cash</span>
          <span className={styles.value}>
            {currency(metrics.cash)}
          </span>
        </div>

        <div className={styles.row}>
          <span className={styles.label}>Unrealized PnL</span>
          <span
            className={
              metrics.unrealized_pnl >= 0
                ? styles.positive
                : styles.negative
            }
          >
            {currency(metrics.unrealized_pnl)}
          </span>
        </div>

        <div className={styles.row}>
          <span className={styles.label}>Realized PnL</span>
          <span
            className={
              metrics.realized_pnl >= 0
                ? styles.positive
                : styles.negative
            }
          >
            {currency(metrics.realized_pnl)}
          </span>
        </div>
      </div>

      <div className={styles.section}>
        <h3>Risk & Trading</h3>

        <div className={styles.highlightRow}>
          <span className={styles.label}>Buying Power</span>
          <span className={styles.highlightValue}>
            {currency(metrics.buying_power)}
          </span>
        </div>

        <div className={styles.row}>
          <span className={styles.label}>
            Maintenance Margin
          </span>
          <span className={styles.value}>
            {currency(metrics.maintenance_margin)}
          </span>
        </div>

        <div className={styles.row}>
          <span className={styles.label}>
            Initial Margin
          </span>
          <span className={styles.value}>
            {currency(metrics.initial_margin)}
          </span>
        </div>

        <div className={styles.row}>
          <span className={styles.label}>
            Account Leverage
          </span>
          <span className={styles.value}>
            {metrics.account_leverage}x
          </span>
        </div>

        <div className={styles.row}>
          <span className={styles.label}>
            Margin Cushion
          </span>
          <span
            className={
              metrics.margin_cushion >= 0
                ? styles.positive
                : styles.negative
            }
          >
            {metrics.margin_cushion.toFixed(2)}%
          </span>
        </div>
      </div>
    </div>
  );
}