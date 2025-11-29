import time
import sys
from .loop_search import start_background_negative_loop_search

def main():
    print("Starting Negative Number Loop Searcher...")
    print("This will search for loops in negative integers (Collatz extension).")
    print("Known cycles (-1, -5, -17) will be ignored.")
    
    # Start search with many threads
    # Start from magnitude 10 (so -10)
    start_background_negative_loop_search(num_threads=20, start_number=10, numbers_per_thread=10000000)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nSearch stopped by user.")

if __name__ == "__main__":
    main()
