import styles from "./evaluation.module.css"
import NavBar from "../components/NavBar/NavBar"
import EvaluationPanel from "../components/EvaluationPanel/EvaluationPanel"

export default function EvaluationPage(){
    return(
      <main>
        <div className={styles.container}>
          <div className={styles.header}>
            <h1 className="Title">Evaluation</h1>
            <NavBar />
          </div>

          <div className={styles.content}>
            <EvaluationPanel></EvaluationPanel>
            <div className={styles.sidebar}>
            </div>
          </div>

          <div className = "footer">
            <p>© 2024 Live Trader Too</p>
          </div>
        </div>
      </main>
    )
}