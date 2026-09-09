import * as React from "react";
import { cn } from "@/lib/utils";

export function Badge({ variant = "default", className, ...props }: React.HTMLAttributes<HTMLSpanElement> & { variant?: "default" | "destructive" }) {
  return <span className={cn("inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium", variant === "destructive" ? "border-red-500/40 bg-red-950/40 text-red-300" : "border-emerald-500/40 bg-emerald-950/40 text-emerald-300", className)} {...props} />;
}
