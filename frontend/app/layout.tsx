import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Live Trader Too",
  description: "Live Trading Dashboard built with Next.js",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}