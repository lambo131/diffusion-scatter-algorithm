import multiprocessing as mp
import time
from pynput import keyboard

def worker(name):
    print(f"{name} started")
    try:
        while True:
            print(f"{name} working...")
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"{name} received termination signal")
    finally:
        print(f"{name} cleaning up...")

def on_press(key, process):
    try:
        if key == keyboard.Key.esc:
            print("\nESC pressed - terminating child process")
            process.terminate()
            process.join()
            return False  # Stop listener
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    process = mp.Process(target=worker, args=("Worker-1",))
    process.start()
    
    print("Main process running. Press ESC to terminate.")
    
    # Set up keyboard listener
    with keyboard.Listener(
        on_press=lambda key: on_press(key, process)
    ) as listener:
        try:
            while process.is_alive():
                time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            listener.stop()
    
    print("Program exited cleanly")