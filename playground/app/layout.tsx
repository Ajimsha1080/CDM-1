import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "../components/Sidebar";
import { Header } from "../components/Header";
import { MobileNavProvider } from "../components/MobileNavContext";

export const metadata: Metadata = {
  title: "CDM-1 Developer Platform",
  description: "Enterprise Cognitive Decision Model & Autonomous Agent Platform.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full bg-[#FAF9F5] text-[#1A1816]">
      <body className="h-full flex antialiased selection:bg-[#EAE6DC] selection:text-[#1A1816] overflow-hidden font-sans bg-[#FAF9F5] text-[#1A1816]">
        <MobileNavProvider>
          <Sidebar />
          <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden bg-[#FAF9F5]">
            <Header />
            <main className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8 bg-[#FAF9F5] text-[#1A1816]">
              {children}
            </main>
          </div>
        </MobileNavProvider>
      </body>
    </html>
  );
}
