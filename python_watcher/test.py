# NOTE: must install the "watchdog" package (available in pip).

from watchdog.events import FileSystemEventHandler


class handler(FileSystemEventHandler):

    def __init__(self, callback, *args, **kwargs):
        super(handler, self).__init__(*args, **kwargs)
        self._callback = callback

    def on_modified(self, event):
        import json
        if event.is_directory:
            return
        try:
            print("Loading JSON file '{}'".format(event.src_path))
            djson = json.load(open(event.src_path))
        except Exception as e:
            print("JSON loading from file '{}' failed.".format(event.src_path))
            print("The exception message:\n{}".format(str(e)))
            return
        try:
            self._callback(djson)
        except Exception as e:
            print("The callback for file '{}' failed.".format(event.src_path))
            print("The exception message:\n{}".format(str(e)))
            return


class json_observer(object):

    def __init__(self, callback, path="."):
        from watchdog.observers import Observer
        self._observer = Observer()
        event_handler = handler(callback)
        self._observer.schedule(event_handler, path, recursive=True)
        self._observer.start()

    def stop(self):
        self._observer.stop()
        self._observer.join()

if __name__ == "__main__":
    import time
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    observer = json_observer(lambda d: print(d), path)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
