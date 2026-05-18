"use client";

/**
 * Page-level Framer Motion transition wrapper.
 * Keyed on pathname so AnimatePresence can swap children with a cross-fade.
 */
import { AnimatePresence, motion } from "framer-motion";
import { usePathname } from "next/navigation";

export function PageTransition({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={pathname}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -8 }}
        transition={{ duration: 0.2, ease: "easeOut" }}
        className="flex-1"
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
