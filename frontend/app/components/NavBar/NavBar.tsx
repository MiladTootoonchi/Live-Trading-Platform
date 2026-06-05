
import Link from "next/link";
import styles from "./NavBar.module.css";


export default function NavBar() {
  return (
    <div className={styles.navBar}>
        <Link href="/" className={styles.navButton}>
            <button>Home</button>
        </Link>
        <Link href="/history" className={styles.navButton}>
            <button>History</button>
        </Link>
        <Link href="/evaluations" className={styles.navButton}>
            <button>Evaluations</button>
        </Link>
        <Link href="/settings" className={styles.navButton}>
            <button>Settings</button>
        </Link>
    </div>
  );
}