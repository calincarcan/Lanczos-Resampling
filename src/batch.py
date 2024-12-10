import os
import sys
import time

def main():
    folder = "../bin/"
    if not os.path.exists(folder):
        print("Folder does not exist")
        sys.exit(1)


    for file in os.listdir(folder):
        for i in range(9, 13):
            start = time.time()
            os.system(f"./{folder}/{file} {2**i} {2**(i+3)}")
            end = time.time()
            print(f"Time taken for {file} with {2**i} and {2**(i+3)} is {end-start}")

if __name__ == "__main__":
    main()
