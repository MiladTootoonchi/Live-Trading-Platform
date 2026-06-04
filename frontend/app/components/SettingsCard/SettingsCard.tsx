type SettingsCardProps = {
  title: string;
  children: React.ReactNode;
};

export default function SettingsCard({title,children,}: SettingsCardProps) {
  return (
    <div className="card">
      <h2>{title}</h2>
      {children}
    </div>
  );
}