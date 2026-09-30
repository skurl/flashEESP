"""Embed a protein sequence on the ESP32, over WiFi or the USB cable.
    python embed.py MQIFVKTL... 172.20.10.2           # WiFi (IP printed on boot)
    python embed.py MQIFVKTL... /dev/cu.usbserial-... # cable in the UART/COM port (close idf.py monitor first)"""
import os, select, socket, sys, termios, tty


def _serial(seq, dev, timeout=600):
    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY)
    try:
        tty.setraw(fd)
        a = termios.tcgetattr(fd); a[4] = a[5] = termios.B115200; termios.tcsetattr(fd, termios.TCSANOW, a)
        termios.tcflush(fd, termios.TCIFLUSH)            # drop boot logs still sitting in the buffer
        os.write(fd, seq.encode() + b"\n")
        buf = b""
        while True:                                      # logs share the console; the answer is the all-numbers line
            if not select.select([fd], [], [], timeout)[0]: raise TimeoutError(buf[-300:].decode(errors="replace"))
            buf += os.read(fd, 4096)
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                if ln.startswith(b"ready:"):             # opening the COM port resets the board: ask again once it's up
                    os.write(fd, seq.encode() + b"\n")
                parts = ln.split()
                if len(parts) > 32:
                    try: return [float(x) for x in parts]
                    except ValueError: pass
    finally:
        os.close(fd)


def embed(seq, host="localhost", port=5000):
    if host.startswith("/dev/"): return _serial(seq, host)
    with socket.create_connection((host, port), timeout=600) as s:   # inference is slow; be patient
        s.sendall(seq.encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(4096)
            if not chunk: raise ConnectionError(buf.decode() or "closed")
            buf += chunk
    return [float(x) for x in buf.split()]


if __name__ == "__main__":
    v = embed(sys.argv[1], *sys.argv[2:3])
    assert len(v) == 320, len(v)
    print(len(v), v[:8])
