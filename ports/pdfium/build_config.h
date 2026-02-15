// Chromium build_config.h shim for standalone PDFium builds.
// Provides platform/compiler/architecture detection macros that PDFium
// expects from the Chromium build system.
#ifndef BUILD_BUILD_CONFIG_H_
#define BUILD_BUILD_CONFIG_H_

#include "build/buildflag.h"

// Compiler detection
#if defined(__GNUC__) || defined(__clang__)
#define COMPILER_GCC 1
#endif
#if defined(_MSC_VER)
#define COMPILER_MSVC 1
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ARCH_CPU_X86_FAMILY 1
#endif
#if defined(__x86_64__) || defined(_M_X64)
#define ARCH_CPU_X86_64 1
#define ARCH_CPU_64_BITS 1
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#define ARCH_CPU_ARM64 1
#define ARCH_CPU_64_BITS 1
#endif
#if defined(__arm__) || defined(_M_ARM)
#define ARCH_CPU_ARMEL 1
#endif

// BUILDFLAG_INTERNAL_* values (0 or 1) for use with BUILDFLAG() macro.
// IMPORTANT: Do NOT define bare IS_XXX aliases — they break BUILDFLAG()
// token pasting (the preprocessor expands IS_XXX before pasting).

#if defined(__linux__) && !defined(__ANDROID__)
#define BUILDFLAG_INTERNAL_IS_LINUX 1
#else
#define BUILDFLAG_INTERNAL_IS_LINUX 0
#endif

#if defined(__APPLE__)
#define BUILDFLAG_INTERNAL_IS_APPLE 1
#include <TargetConditionals.h>
#if TARGET_OS_MAC && !TARGET_OS_IPHONE
#define BUILDFLAG_INTERNAL_IS_MAC 1
#else
#define BUILDFLAG_INTERNAL_IS_MAC 0
#endif
#if TARGET_OS_IPHONE
#define BUILDFLAG_INTERNAL_IS_IOS 1
#else
#define BUILDFLAG_INTERNAL_IS_IOS 0
#endif
#else
#define BUILDFLAG_INTERNAL_IS_APPLE 0
#define BUILDFLAG_INTERNAL_IS_MAC 0
#define BUILDFLAG_INTERNAL_IS_IOS 0
#endif

#if defined(_WIN32)
#define BUILDFLAG_INTERNAL_IS_WIN 1
#else
#define BUILDFLAG_INTERNAL_IS_WIN 0
#endif

#if defined(__ANDROID__)
#define BUILDFLAG_INTERNAL_IS_ANDROID 1
#else
#define BUILDFLAG_INTERNAL_IS_ANDROID 0
#endif

#if defined(__linux__) || defined(__APPLE__) || defined(__ANDROID__)
#define BUILDFLAG_INTERNAL_IS_POSIX 1
#else
#define BUILDFLAG_INTERNAL_IS_POSIX 0
#endif

#define BUILDFLAG_INTERNAL_IS_FUCHSIA 0
#define BUILDFLAG_INTERNAL_IS_CHROMEOS 0
#define BUILDFLAG_INTERNAL_IS_PDF_XFA 0

// wchar_t size detection
#if defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#define WCHAR_T_IS_32_BIT
#define WCHAR_T_IS_UTF32
#elif defined(_WIN32)
#define WCHAR_T_IS_16_BIT
#define WCHAR_T_IS_UTF16
#endif

#endif  // BUILD_BUILD_CONFIG_H_
