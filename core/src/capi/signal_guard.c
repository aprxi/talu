/**
 * Signal Guard - C implementation for sigsetjmp/siglongjmp
 *
 * This provides async-signal-safe recovery from SIGBUS using sigsetjmp/siglongjmp.
 * sigsetjmp/siglongjmp are macros in C, so we need this C wrapper.
 *
 * The key insight is that sigsetjmp/siglongjmp work like this:
 * - sigsetjmp saves the execution context (registers, stack pointer, etc.)
 * - siglongjmp restores that context and makes sigsetjmp return again
 *
 * The problem with wrapping sigsetjmp in a function is that when siglongjmp
 * jumps back, the function returns normally - but the caller has already moved
 * past the call site!
 *
 * Solution: Use a callback pattern where the protected code is passed as a
 * function pointer. This way, sigsetjmp and the check happen in the same
 * function, and siglongjmp correctly skips the callback.
 */

#define _POSIX_C_SOURCE 200809L
#include <signal.h>
#include <setjmp.h>
#include <stdatomic.h>
#include <stddef.h>

/* Thread-local jump buffer and state */
static _Thread_local sigjmp_buf talu_signal_guard_jmpbuf;
static _Thread_local volatile sig_atomic_t talu_signal_guard_active = 0;

/* Signal handler - jumps back to setjmp point if guard is active */
static void talu_signal_handler(int sig) {
    if (talu_signal_guard_active) {
        talu_signal_guard_active = 0;
        siglongjmp(talu_signal_guard_jmpbuf, 1);
    }
    /* If guard not active, re-raise with default handler */
    signal(sig, SIG_DFL);
    raise(sig);
}

/* Install signal handlers - call once per thread */
void talu_signal_guard_install(void) {
    struct sigaction sa;
    sa.sa_handler = talu_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);  /* Also catch SIGSEGV for memory access errors */
}

/*
 * Execute a function with SIGBUS protection.
 *
 * @param func The function to execute (takes void* context, returns int error code)
 * @param ctx  Context pointer passed to func
 * @return 0 if func completed successfully (func returned 0)
 *         func's return value if func returned non-zero
 *         -1 if SIGBUS was caught during execution
 */
int talu_signal_guard_call(int (*func)(void*), void* ctx) {
    if (sigsetjmp(talu_signal_guard_jmpbuf, 1) == 0) {
        talu_signal_guard_active = 1;
        int result = func(ctx);
        talu_signal_guard_active = 0;
        return result;
    } else {
        /* siglongjmp jumped here - signal was caught */
        return -1;
    }
}
