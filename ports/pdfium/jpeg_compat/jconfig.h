// Minimal jconfig.h for PDFium compilation against libjpeg-turbo headers.
// Only used at compile time; actual libjpeg-turbo is linked by build.zig.
#define JPEG_LIB_VERSION  62
#define LIBJPEG_TURBO_VERSION  3.1.3
#define LIBJPEG_TURBO_VERSION_NUMBER  3001003
#define C_ARITH_CODING_SUPPORTED 1
#define D_ARITH_CODING_SUPPORTED 1
#define MEM_SRCDST_SUPPORTED  1
#ifndef BITS_IN_JSAMPLE
#define BITS_IN_JSAMPLE  8
#endif
#ifndef _WIN32
/* #undef RIGHT_SHIFT_IS_UNSIGNED */
#else
#undef RIGHT_SHIFT_IS_UNSIGNED
#ifndef __RPCNDR_H__
typedef unsigned char boolean;
#endif
#define HAVE_BOOLEAN
#if !(defined(_BASETSD_H_) || defined(_BASETSD_H))
typedef short INT16;
typedef signed int INT32;
#endif
#define XMD_H
#endif
