import styles from "./IsMarketOpen.module.css";

export default function IsMarketOpen({isOpen}: {isOpen: boolean}) {
  return (
    <div className={styles.marketStatus}>
      {isOpen ? "Market is Open" : "Market is Closed"}
    </div>
  );
}