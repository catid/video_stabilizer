#include "tools.hpp"

#if defined(_WIN32) || defined(_WIN64)
    #define IS_WINDOWS
    #include <windows.h>
#else
    #include <ctime>
#endif

uint64_t get_time_since_boot_microseconds() {
#ifdef IS_WINDOWS
    // Windows implementation using GetTickCount64
    // GetTickCount64 returns milliseconds since system boot
    ULONGLONG milliseconds = GetTickCount64();
    
    // Convert milliseconds to microseconds
    return static_cast<uint64_t>(milliseconds) * 1000;
#else
    // Linux implementation using clock_gettime
    struct timespec ts;
    
    // CLOCK_BOOTTIME includes time spent in suspend
    // If CLOCK_BOOTTIME is not available, use CLOCK_MONOTONIC
    #ifdef CLOCK_BOOTTIME
        if (clock_gettime(CLOCK_BOOTTIME, &ts) != 0) {
            // Handle error
            return 0;
        }
    #else
        if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
            // Handle error
            return 0;
        }
    #endif
    
    // Convert seconds and nanoseconds to microseconds
    uint64_t microseconds = static_cast<uint64_t>(ts.tv_sec) * 1000000
                             + static_cast<uint64_t>(ts.tv_nsec) / 1000;
    return microseconds;
#endif
}