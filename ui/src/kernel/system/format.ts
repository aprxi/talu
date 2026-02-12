/** Standardized formatters for dates, numbers, durations. */

import type { FormatAccess } from "../types.ts";

const SHORT_DATE: Intl.DateTimeFormatOptions = { month: "short", day: "numeric" };
const MEDIUM_DATE: Intl.DateTimeFormatOptions = { month: "short", day: "numeric", year: "numeric" };
const LONG_DATE: Intl.DateTimeFormatOptions = { weekday: "long", month: "long", day: "numeric", year: "numeric" };

const SHORT_DATETIME: Intl.DateTimeFormatOptions = { ...SHORT_DATE, hour: "2-digit", minute: "2-digit" };
const MEDIUM_DATETIME: Intl.DateTimeFormatOptions = { ...MEDIUM_DATE, hour: "2-digit", minute: "2-digit" };
const LONG_DATETIME: Intl.DateTimeFormatOptions = { ...LONG_DATE, hour: "2-digit", minute: "2-digit" };

const STYLES: Record<string, Intl.DateTimeFormatOptions> = {
  short: SHORT_DATE,
  medium: MEDIUM_DATE,
  long: LONG_DATE,
};

const DATETIME_STYLES: Record<string, Intl.DateTimeFormatOptions> = {
  short: SHORT_DATETIME,
  medium: MEDIUM_DATETIME,
  long: LONG_DATETIME,
};

const UNITS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["year", 365 * 24 * 60 * 60],
  ["month", 30 * 24 * 60 * 60],
  ["week", 7 * 24 * 60 * 60],
  ["day", 24 * 60 * 60],
  ["hour", 60 * 60],
  ["minute", 60],
  ["second", 1],
];

export class FormatAccessImpl implements FormatAccess {
  date(value: Date | string | number, style: "short" | "medium" | "long" = "medium"): string {
    const d = value instanceof Date ? value : new Date(value);
    return new Intl.DateTimeFormat(undefined, STYLES[style]).format(d);
  }

  dateTime(value: Date | string | number, style: "short" | "medium" | "long" = "medium"): string {
    const d = value instanceof Date ? value : new Date(value);
    return new Intl.DateTimeFormat(undefined, DATETIME_STYLES[style]).format(d);
  }

  number(value: number, options?: Intl.NumberFormatOptions): string {
    return new Intl.NumberFormat(undefined, options).format(value);
  }

  relativeTime(value: Date | string | number): string {
    const d = value instanceof Date ? value : new Date(value);
    const diff = (d.getTime() - Date.now()) / 1000;
    const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

    for (const [unit, secs] of UNITS) {
      if (Math.abs(diff) >= secs) {
        return rtf.format(Math.round(diff / secs), unit);
      }
    }
    return rtf.format(0, "second");
  }

  duration(seconds: number): string {
    const abs = Math.abs(Math.floor(seconds));
    const h = Math.floor(abs / 3600);
    const m = Math.floor((abs % 3600) / 60);
    const s = abs % 60;

    const parts: string[] = [];
    if (h > 0) parts.push(`${h}h`);
    if (m > 0) parts.push(`${m}m`);
    if (s > 0 || parts.length === 0) parts.push(`${s}s`);
    return parts.join(" ");
  }
}
