import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ARGUS Stream A | Video Anomaly Detection",
  description:
    "Unsupervised frame-level video anomaly detection with Avenue benchmarks, Modal GPU inference, and a Vercel frontend.",
  openGraph: {
    title: "ARGUS Stream A | Video Anomaly Detection",
    description:
      "Unsupervised frame-level anomaly detection with VideoMAE features, MULDE scoring, Modal GPU inference, and Vercel deployment.",
    url: "https://anamolydetect.vercel.app",
    siteName: "ARGUS Stream A",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
