import { ComparisonLab } from "@/components/tabs/ComparisonLab";

export const metadata = {
  title: "Compare · Atlas Raman",
  description:
    "Stage 4–12+ spectra, line up axes, pick differences — small multiples, overlay, waterfall, heatmap, and diff-vs-reference views.",
};

export default function ComparePage() {
  return (
    <div className="min-h-screen px-6 md:px-10 py-8 md:py-10">
      <ComparisonLab />
    </div>
  );
}
