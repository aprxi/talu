// ICU function shim for PDFium.
//
// PDFium compiled with USE_SYSTEM_ICUUC references versioned ICU symbols
// (e.g. u_tolower_74). Rather than linking the full ICU library (~34MB),
// we provide the 7 functions PDFium actually uses, backed by utf8proc's
// Unicode Character Database for full Unicode correctness across all scripts.
//
// ICU version suffix must match what PDFium was compiled against.

#include <stdint.h>
#include "utf8proc.h"

// ICU type aliases
typedef int32_t UChar32;
typedef int8_t UBool;

UChar32 u_tolower_74(UChar32 c) {
    return (UChar32)utf8proc_tolower((utf8proc_int32_t)c);
}

UChar32 u_toupper_74(UChar32 c) {
    return (UChar32)utf8proc_toupper((utf8proc_int32_t)c);
}

UBool u_isalpha_74(UChar32 c) {
    utf8proc_category_t cat = utf8proc_category((utf8proc_int32_t)c);
    // Unicode General Category L* (Lu, Ll, Lt, Lm, Lo)
    return (cat >= UTF8PROC_CATEGORY_LU && cat <= UTF8PROC_CATEGORY_LO) ? 1 : 0;
}

UBool u_isalnum_74(UChar32 c) {
    utf8proc_category_t cat = utf8proc_category((utf8proc_int32_t)c);
    // L* (letters) or N* (numbers: Nd, Nl, No)
    return ((cat >= UTF8PROC_CATEGORY_LU && cat <= UTF8PROC_CATEGORY_LO) ||
            (cat >= UTF8PROC_CATEGORY_ND && cat <= UTF8PROC_CATEGORY_NO)) ? 1 : 0;
}

UBool u_isspace_74(UChar32 c) {
    // Match ICU's u_isspace: Z* separators + whitespace controls
    if (c == 0x09 || c == 0x0A || c == 0x0B || c == 0x0C || c == 0x0D ||
        c == 0x1C || c == 0x1D || c == 0x1E || c == 0x1F || c == 0x85) {
        return 1;
    }
    utf8proc_category_t cat = utf8proc_category((utf8proc_int32_t)c);
    // Zs (space separator), Zl (line separator), Zp (paragraph separator)
    return (cat >= UTF8PROC_CATEGORY_ZS && cat <= UTF8PROC_CATEGORY_ZP) ? 1 : 0;
}

UBool u_islower_74(UChar32 c) {
    return utf8proc_category((utf8proc_int32_t)c) == UTF8PROC_CATEGORY_LL ? 1 : 0;
}

UBool u_isupper_74(UChar32 c) {
    return utf8proc_category((utf8proc_int32_t)c) == UTF8PROC_CATEGORY_LU ? 1 : 0;
}
