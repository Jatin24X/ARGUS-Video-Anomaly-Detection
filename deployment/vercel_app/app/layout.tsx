import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ARGUS | Unsupervised Video Anomaly Detection",
  description:
    "Unsupervised video anomaly detection dashboard utilizing spatio-temporal Vision Transformers (VideoMAE-v2), multiscale density estimators, and scale-to-zero serverless GPU pipelines.",
  openGraph: {
    title: "ARGUS | Unsupervised Video Anomaly Detection",
    description:
      "Unsupervised video anomaly detection dashboard utilizing VideoMAE-v2 features, MULDE density scoring, and scale-to-zero serverless GPU inference.",
    url: "https://anamolydetect.vercel.app",
    siteName: "ARGUS",
    type: "website",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "ARGUS Dashboard Preview",
      },
    ],
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
