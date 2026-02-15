// Chromium BUILDFLAG() shim for standalone PDFium builds.
// Expands BUILDFLAG(X) to BUILDFLAG_INTERNAL_X (0 or 1) via token pasting.
// The BUILDFLAG_INTERNAL_* values are defined in build_config.h.
#ifndef BUILD_BUILDFLAG_H_
#define BUILD_BUILDFLAG_H_

#define BUILDFLAG_CAT_(a, b) a ## b
#define BUILDFLAG(flag) (BUILDFLAG_CAT_(BUILDFLAG_INTERNAL_, flag))

#endif  // BUILD_BUILDFLAG_H_
