// Minimal ICU utypes.h stub for PDFium compilation.
// Only declares the types PDFium actually uses.
#ifndef ICU_STUB_UTYPES_H
#define ICU_STUB_UTYPES_H

#include <stdint.h>

typedef int32_t UChar32;
typedef int8_t UBool;
typedef uint16_t UChar;

#define U_CAPI extern
#define U_STABLE extern
#define U_ICU_ENTRY_POINT_RENAME(x) x ## _74

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

#endif
