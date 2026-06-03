export default function LiveButton({liveRunning, loadingLive, toggleLive,}: 
    {liveRunning: boolean; loadingLive: boolean; toggleLive: () => void;}) {
  return (
    <button
        className={`live-button ${liveRunning ? "stop" : "start"}`}
        onClick={toggleLive}
        disabled={loadingLive}
        >
        {loadingLive
            ? "Loading..."
            : liveRunning
            ? "Stop Live"
            : "Start Live"}
    </button>
  );
}