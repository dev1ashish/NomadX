import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Standard shadcn-style className helper. Re-exported so tab workers can
 *  import from `@/lib/cn` or `@/lib/utils` interchangeably. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
