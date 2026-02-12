/**
 * Pure C test for signal guard SIGBUS recovery.
 *
 * Run via CI script:
 *   ./scripts/check_signal_tests.sh
 *
 * Or manually:
 *   gcc -o signal_guard_test_c core/tests/helpers/signal/signal_guard_test.c core/src/capi/signal_guard.c
 *   ./signal_guard_test_c
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

/* Import from signal_guard.c */
extern void talu_signal_guard_install(void);
extern int talu_signal_guard_call(int (*func)(void*), void* ctx);

/* Test callbacks */
static int normal_callback(void* ctx) {
    (void)ctx;
    /* Just return success */
    return 0;
}

static int sigbus_callback(void* ctx) {
    (void)ctx;
    /* Raise SIGBUS - should be caught */
    raise(SIGBUS);
    /* Should NOT reach here */
    return 0;
}

static int returns_42(void* ctx) {
    (void)ctx;
    return 42;
}

int main(void) {
    talu_signal_guard_install();

    /* Test 1: Normal operation */
    printf("Test 1: Normal operation... ");
    fflush(stdout);
    {
        int result = talu_signal_guard_call(normal_callback, NULL);
        if (result == 0) {
            printf("PASS\n");
        } else {
            printf("FAIL (expected 0, got %d)\n", result);
            return 1;
        }
    }

    /* Test 2: SIGBUS recovery */
    printf("Test 2: SIGBUS recovery... ");
    fflush(stdout);
    {
        int result = talu_signal_guard_call(sigbus_callback, NULL);
        if (result == -1) {
            printf("PASS\n");
        } else {
            printf("FAIL (expected -1, got %d)\n", result);
            return 1;
        }
    }

    /* Test 3: Multiple cycles */
    printf("Test 3: Multiple cycles... ");
    fflush(stdout);
    {
        /* Cycle 1 - normal */
        int result = talu_signal_guard_call(normal_callback, NULL);
        if (result != 0) {
            printf("FAIL (cycle 1: expected 0, got %d)\n", result);
            return 1;
        }

        /* Cycle 2 - with signal */
        result = talu_signal_guard_call(sigbus_callback, NULL);
        if (result != -1) {
            printf("FAIL (cycle 2: expected -1, got %d)\n", result);
            return 1;
        }

        /* Cycle 3 - normal again */
        result = talu_signal_guard_call(normal_callback, NULL);
        if (result != 0) {
            printf("FAIL (cycle 3: expected 0, got %d)\n", result);
            return 1;
        }

        printf("PASS\n");
    }

    /* Test 4: Return value propagation */
    printf("Test 4: Return value propagation... ");
    fflush(stdout);
    {
        int result = talu_signal_guard_call(returns_42, NULL);
        if (result == 42) {
            printf("PASS\n");
        } else {
            printf("FAIL (expected 42, got %d)\n", result);
            return 1;
        }
    }

    printf("\nAll tests passed!\n");
    return 0;
}
