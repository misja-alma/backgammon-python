import queue
import threading


class Logger:
    """Routes status messages to the on-screen text area.

    log() may be called from any thread (e.g. the analysis worker). Messages logged off the
    main thread are queued and only written to the text area when the main thread calls
    flush(), because pygame UI state must only be touched from the main thread.
    """

    _text_area = None
    _pending: queue.SimpleQueue[str] = queue.SimpleQueue()

    @classmethod
    def set_text_area(cls, text_area) -> None:
        cls._text_area = text_area

    @classmethod
    def log(cls, message: str) -> None:
        if cls._text_area is None:
            print(message)
        elif threading.current_thread() is threading.main_thread():
            cls._text_area.show_text(message)
        else:
            cls._pending.put(message)

    @classmethod
    def flush(cls) -> None:
        """Write queued messages to the text area. Call from the main thread only."""
        while True:
            try:
                message = cls._pending.get_nowait()
            except queue.Empty:
                return
            if cls._text_area is None:
                print(message)
            else:
                cls._text_area.show_text(message)
