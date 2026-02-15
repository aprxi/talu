// Minimal ICU uchar.h stub for PDFium compilation.
// Declares only the 7 functions PDFium uses, with _74 versioned suffixes.
#ifndef ICU_STUB_UCHAR_H
#define ICU_STUB_UCHAR_H

#include "unicode/utypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Versioned symbol declarations — match icu_shim.c
UChar32 u_tolower_74(UChar32 c);
UChar32 u_toupper_74(UChar32 c);
UBool u_isalpha_74(UChar32 c);
UBool u_isalnum_74(UChar32 c);
UBool u_isspace_74(UChar32 c);
UBool u_islower_74(UChar32 c);
UBool u_isupper_74(UChar32 c);

// Rename macros (same as ICU's urename.h)
#define u_tolower u_tolower_74
#define u_toupper u_toupper_74
#define u_isalpha u_isalpha_74
#define u_isalnum u_isalnum_74
#define u_isspace u_isspace_74
#define u_islower u_islower_74
#define u_isupper u_isupper_74

#ifdef __cplusplus
}
#endif

#endif
