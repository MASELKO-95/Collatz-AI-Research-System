#include <cstdint>
#include <vector>
#include <random>
#include <iostream>

// Use __int128 for numbers > 2^64
// GCC/Clang support this extension
typedef unsigned __int128 uint128_t;

extern "C" {

    // Helper to check if a number is in a cycle (Floyd's algorithm)
    // Returns: 0 = converged to 1, 1 = non-trivial cycle, 2 = max steps reached/overflow
    int check_cycle(uint64_t n_high, uint64_t n_low, int max_steps) {
        uint128_t n = ((uint128_t)n_high << 64) | n_low;
        uint128_t tortoise = n;
        uint128_t hare = n;
        int steps = 0;

        while (steps < max_steps) {
            if (tortoise == 1 || hare == 1) return 0;

            // Tortoise step
            if (tortoise % 2 == 0) tortoise /= 2;
            else {
                if (tortoise > (((uint128_t)-1) - 1) / 3) return 2; // Overflow check
                tortoise = 3 * tortoise + 1;
            }

            // Hare step 1
            if (hare % 2 == 0) hare /= 2;
            else {
                if (hare > (((uint128_t)-1) - 1) / 3) return 2;
                hare = 3 * hare + 1;
            }
            if (hare == 1) return 0;

            // Hare step 2
            if (hare % 2 == 0) hare /= 2;
            else {
                if (hare > (((uint128_t)-1) - 1) / 3) return 2;
                hare = 3 * hare + 1;
            }

            if (tortoise == hare) {
                // Trivial cycle 1-4-2-1
                if (tortoise == 1 || tortoise == 2 || tortoise == 4) return 0;
                return 1; // Found non-trivial cycle
            }
            steps++;
        }
        return 2; // Max steps
    }

    // Generate a batch of hard candidates
    // out_nums: array of 2 * count uint64 (high, low pairs)
    // out_stops: array of count uint16
    // out_parity: array of count * max_len int8
    void generate_hard_batch(
        int count, 
        int max_len,
        uint64_t* out_nums, 
        int32_t* out_stops, 
        int8_t* out_parity
    ) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dist_u64;
        
        uint128_t base_2_68 = (uint128_t)1 << 68;

        for (int i = 0; i < count; ++i) {
            // Strategy selection
            int strategy = dist_u64(gen) % 10;
            uint128_t n;

            if (strategy < 3) {
                // Random large > 2^68
                uint64_t offset = dist_u64(gen) % 1000000000000ULL;
                n = base_2_68 + offset;
            } else if (strategy < 6) {
                // 3 mod 4
                uint64_t offset = dist_u64(gen) % 1000000000000ULL;
                n = base_2_68 + offset;
                if (n % 4 != 3) n += (3 - (n % 4));
            } else {
                // Dense 1s (heuristic)
                n = 0;
                int len = 70 + (dist_u64(gen) % 50);
                for(int b=0; b<len; ++b) {
                    n <<= 1;
                    if ((dist_u64(gen) % 10) < 8) n |= 1;
                }
                if (n < base_2_68) n += base_2_68;
            }

            // Store number
            out_nums[2*i] = (uint64_t)(n >> 64);
            out_nums[2*i+1] = (uint64_t)(n & 0xFFFFFFFFFFFFFFFFULL);

            // Calculate sequence
            uint128_t curr = n;
            int steps = 0;
            
            while (curr > 1 && steps < max_len) {
                if (curr % 2 == 0) {
                    out_parity[i * max_len + steps] = 0;
                    curr /= 2;
                } else {
                    out_parity[i * max_len + steps] = 1;
                    // Check overflow? 128 bit is huge, unlikely to overflow for these inputs in 500 steps
                    curr = 3 * curr + 1;
                }
                steps++;
            }
            
            // Fill remaining parity with -1
            for (int k = steps; k < max_len; ++k) {
                out_parity[i * max_len + k] = -1;
            }

            // Continue for stopping time
            int total_steps = steps;
            while (curr > 1 && total_steps < 20000) {
                 if (curr % 2 == 0) curr /= 2;
                 else curr = 3 * curr + 1;
                 total_steps++;
            }
            out_stops[i] = total_steps;
        }
    }
}
