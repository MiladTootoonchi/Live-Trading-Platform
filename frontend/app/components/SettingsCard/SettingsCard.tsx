import styles from "./SettingsCard.module.css"

type SettingsCardProps = {
  title: string;
  children: React.ReactNode;
};

export default function SettingsCard({title,children,}: SettingsCardProps) {
  return (
    <div className={styles.card}>
      <h2 className={styles.settingsTitle}>{title}</h2>
      {children}
    </div>
  );
}