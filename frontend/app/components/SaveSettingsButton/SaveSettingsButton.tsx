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
      onClick={onClick}
      disabled={loading}
    >
      {loading ? "Saving..." : "Save Changes"}
    </button>
  );
}