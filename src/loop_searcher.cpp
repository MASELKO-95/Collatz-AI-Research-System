#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

typedef unsigned __int128 uint128_t;

std::atomic<bool> loop_found(false);
std::mutex output_mutex;

// Check for cycle in range
void search_range(uint64_t start_high, uint64_t start_low, uint64_t count,
                  int thread_id) {
  std::random_device rd;
  std::mt19937_64 gen(rd() + thread_id);
  std::uniform_int_distribution<uint64_t> dist;

  uint128_t base = ((uint128_t)start_high << 64) | start_low;

  for (uint64_t i = 0; i < count && !loop_found.load(); ++i) {
    // Generate candidate
    uint128_t n;

    // Strategy: focus on n > 2^68, n ≡ 3 (mod 4)
    if (i % 2 == 0) {
      // Sequential search from base
      n = base + i;
    } else {
      // Random large number
      uint64_t offset = dist(gen);
      n = base + offset;
    }

    // Ensure n ≡ 3 (mod 4)
    if (n % 4 != 3) {
      n += (3 - (n % 4));
    }

    // Floyd's cycle detection
    uint128_t tortoise = n;
    uint128_t hare = n;
    int steps = 0;
    const int MAX_STEPS = 100000;

    while (steps < MAX_STEPS && !loop_found.load()) {
      // Tortoise step
      if (tortoise == 1)
        break;
      if (tortoise % 2 == 0) {
        tortoise /= 2;
      } else {
        if (tortoise > (((uint128_t)-1) - 1) / 3)
          break; // Overflow
        tortoise = 3 * tortoise + 1;
      }

      // Hare step 1
      if (hare == 1)
        break;
      if (hare % 2 == 0) {
        hare /= 2;
      } else {
        if (hare > (((uint128_t)-1) - 1) / 3)
          break;
        hare = 3 * hare + 1;
      }

      // Hare step 2
      if (hare == 1)
        break;
      if (hare % 2 == 0) {
        hare /= 2;
      } else {
        if (hare > (((uint128_t)-1) - 1) / 3)
          break;
        hare = 3 * hare + 1;
      }

      // Check for cycle
      if (tortoise == hare) {
        // Trivial cycle check
        if (tortoise == 1 || tortoise == 2 || tortoise == 4) {
          break; // Normal convergence
        }

        // NON-TRIVIAL CYCLE FOUND!
        loop_found.store(true);

        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "\n🚨🚨🚨 NON-TRIVIAL CYCLE DETECTED! 🚨🚨🚨\n";
        std::cout << "Thread " << thread_id << " found cycle!\n";
        std::cout << "Starting number (high, low): " << (uint64_t)(n >> 64)
                  << ", " << (uint64_t)(n & 0xFFFFFFFFFFFFFFFFULL) << "\n";
        std::cout << "Cycle value (high, low): " << (uint64_t)(tortoise >> 64)
                  << ", " << (uint64_t)(tortoise & 0xFFFFFFFFFFFFFFFFULL)
                  << "\n";
        return;
      }

      steps++;
    }

    // Progress report every 10000 numbers
    if (i % 10000 == 0 && thread_id == 0) {
      std::lock_guard<std::mutex> lock(output_mutex);
      std::cout << "Thread " << thread_id << " checked " << i
                << " numbers...\n";
    }
  }
}

// Check for cycle in negative range
void search_range_negative(uint64_t start_high, uint64_t start_low,
                           uint64_t count, int thread_id) {
  std::random_device rd;
  std::mt19937_64 gen(rd() + thread_id);
  std::uniform_int_distribution<uint64_t> dist;

  // Construct base as negative number
  // We treat the input as magnitude, so actual number is -(base + i)
  uint128_t base_mag = ((uint128_t)start_high << 64) | start_low;

  for (uint64_t i = 0; i < count && !loop_found.load(); ++i) {
    // Generate candidate magnitude
    uint128_t mag;

    if (i % 2 == 0) {
      mag = base_mag + i;
    } else {
      uint64_t offset = dist(gen);
      mag = base_mag + offset;
    }

    // n is negative
    __int128 n = -(__int128)mag;

    // Skip if n is even (immediately divides by 2)
    if (n % 2 == 0)
      continue;

    // Floyd's cycle detection
    __int128 tortoise = n;
    __int128 hare = n;
    int steps = 0;
    const int MAX_STEPS = 100000;

    while (steps < MAX_STEPS && !loop_found.load()) {
      // Helper lambda for Collatz step on negative numbers
      auto step_func = [](__int128 val) -> __int128 {
        if (val % 2 == 0) {
          return val / 2;
        } else {
          // Check for underflow (large negative * 3)
          // limit is roughly min_int / 3
          // if val < -5e37... break
          return 3 * val + 1;
        }
      };

      // Tortoise step
      tortoise = step_func(tortoise);
      if (tortoise == -1 || tortoise == -5 || tortoise == -17)
        break; // Known cycles

      // Hare step 1
      hare = step_func(hare);
      if (hare == -1 || hare == -5 || hare == -17)
        break;

      // Hare step 2
      hare = step_func(hare);
      if (hare == -1 || hare == -5 || hare == -17)
        break;

      // Check for cycle
      if (tortoise == hare) {
        // Check if it's one of the known cycles (just in case)
        if (tortoise == -1 || tortoise == -5 || tortoise == -17)
          break;

        // NON-TRIVIAL NEW CYCLE FOUND!
        loop_found.store(true);

        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << "\n🚨🚨🚨 NEW NEGATIVE CYCLE DETECTED! 🚨🚨🚨\n";
        std::cout << "Thread " << thread_id << " found cycle!\n";
        // Print as magnitude for easier reading, or cast to int64 parts
        // Note: printing __int128 is tricky, we'll just print high/low of
        // magnitude
        uint128_t start_mag = (n < 0) ? -n : n;
        uint128_t cycle_mag = (tortoise < 0) ? -tortoise : tortoise;

        std::cout << "Starting number: -" << (uint64_t)(start_mag >> 64)
                  << "..." << (uint64_t)(start_mag & 0xFFFFFFFFFFFFFFFFULL)
                  << "\n";
        std::cout << "Cycle value: -" << (uint64_t)(cycle_mag >> 64) << "..."
                  << (uint64_t)(cycle_mag & 0xFFFFFFFFFFFFFFFFULL) << "\n";
        return;
      }

      steps++;
    }

    if (i % 50000 == 0 && thread_id == 0) {
      std::lock_guard<std::mutex> lock(output_mutex);
      std::cout << "Thread " << thread_id << " checked " << i
                << " negative numbers...\n";
    }
  }
}

extern "C" {
// Launch parallel search
// num_threads: number of parallel threads
// start_high, start_low: starting number (128-bit split)
// numbers_per_thread: how many numbers each thread checks
void parallel_loop_search(int num_threads, uint64_t start_high,
                          uint64_t start_low, uint64_t numbers_per_thread) {
  std::vector<std::thread> threads;

  std::cout << "🔍 Starting parallel loop search with " << num_threads
            << " threads...\n";
  std::cout << "Base number: " << start_high << " (high), " << start_low
            << " (low)\n";
  std::cout << "Each thread will check " << numbers_per_thread << " numbers.\n";

  loop_found.store(false);

  for (int i = 0; i < num_threads; ++i) {
    // Each thread gets a different starting offset
    uint64_t offset = i * numbers_per_thread;
    uint64_t thread_start_low = start_low + offset;
    uint64_t thread_start_high = start_high;

    // Handle overflow
    if (thread_start_low < start_low) {
      thread_start_high++;
    }

    threads.emplace_back(search_range, thread_start_high, thread_start_low,
                         numbers_per_thread, i);
  }

  // Wait for all threads
  for (auto &t : threads) {
    t.join();
  }

  if (!loop_found.load()) {
    std::cout
        << "Search completed. No non-trivial cycles found in this range.\n";
  }
}

void parallel_loop_search_negative(int num_threads, uint64_t start_high,
                                   uint64_t start_low,
                                   uint64_t numbers_per_thread) {
  std::vector<std::thread> threads;

  std::cout << "🔍 Starting parallel NEGATIVE loop search with " << num_threads
            << " threads...\n";
  std::cout << "Base magnitude: " << start_high << " (high), " << start_low
            << " (low)\n";

  loop_found.store(false);

  for (int i = 0; i < num_threads; ++i) {
    uint64_t offset = i * numbers_per_thread;
    uint64_t thread_start_low = start_low + offset;
    uint64_t thread_start_high = start_high;

    if (thread_start_low < start_low) {
      thread_start_high++;
    }

    threads.emplace_back(search_range_negative, thread_start_high,
                         thread_start_low, numbers_per_thread, i);
  }

  for (auto &t : threads) {
    t.join();
  }

  if (!loop_found.load()) {
    std::cout << "Negative search completed. No new cycles found.\n";
  }
}
}
