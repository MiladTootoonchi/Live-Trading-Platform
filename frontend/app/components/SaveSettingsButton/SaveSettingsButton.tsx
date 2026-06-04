import styles from "./SaveSettingsButton.module.css"

type SaveSettingsButtonProps = {
  onClick: () => void;
  loading: boolean;
};

export default function SaveSettingsButton({
  onClick,
  loading,
}: SaveSettingsButtonProps) {
  return (
    <button
        className={styles.saveButton}
        onClick={onClick}
        disabled={loading}
    >
        {loading ? "Saving..." : "Save Changes"}
    </button>
  );
}