import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Head to Head — baseline",
};

export default function H2HLayout({ children }: { children: React.ReactNode }) {
  return children;
}
