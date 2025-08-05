import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import os

class DataChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".csv"):
            print(f"[INFO] New data detected: {event.src_path}")
            print("[INFO] Retraining model...")
            subprocess.call(["python", "src/train.py"])
            print("[INFO] Model retraining complete!")

if __name__ == "__main__":
    path = "data/raw/"
    if not os.path.exists(path):
        os.makedirs(path)

    event_handler = DataChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()
    print(f"[INFO] Watching for new data in '{path}'...")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
