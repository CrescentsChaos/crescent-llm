from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import shutil
import webbrowser
import threading

PROJECT_ROOT = Path(__file__).resolve().parent
HTML_FILE = PROJECT_ROOT / "data_entry.html"
TRAINING_FILE = PROJECT_ROOT / "data" / "raw" / "training.txt"
BACKUP_FILE = PROJECT_ROOT / "data" / "raw" / "training.backup.txt"
HOST = "127.0.0.1"
PORT = 8000


# ----------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------

def normalize(line):
    """Collapse every run of whitespace (including stray tabs and
    newlines) down to a single space, and trim the ends."""
    return " ".join(line.split())


def dedupe_key(line):
    """Two conversations count as the same if they only differ by
    whitespace or letter case."""
    return normalize(line).lower()


def read_lines():
    if not TRAINING_FILE.exists():
        return []
    text = TRAINING_FILE.read_text(encoding="utf-8")
    return text.splitlines()


def is_malformed(line):
    return not (
        "<USER>" in line
        and "<ASSISTANT>" in line
        and line.rstrip().endswith("<END>")
    )


def clean(lines):
    """Return (cleaned_lines, stats).

    Blank lines are dropped, internal whitespace is normalized, and
    exact duplicates are removed keeping the first occurrence so the
    original ordering survives.
    """
    blank_removed = 0
    duplicates_removed = 0
    whitespace_fixed = 0

    seen = set()
    cleaned = []

    for line in lines:
        if not line.strip():
            blank_removed += 1
            continue

        normalized = normalize(line)
        if normalized != line:
            whitespace_fixed += 1

        key = dedupe_key(normalized)
        if key in seen:
            duplicates_removed += 1
            continue

        seen.add(key)
        cleaned.append(normalized)

    stats = {
        "blank_removed": blank_removed,
        "duplicates_removed": duplicates_removed,
        "whitespace_fixed": whitespace_fixed,
        "before": len(lines),
        "after": len(cleaned),
        "malformed": sum(1 for line in cleaned if is_malformed(line)),
    }
    return cleaned, stats


def write_lines(lines):
    TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Always end with a newline so the next append starts a new line.
    TRAINING_FILE.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


# ----------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # ------------------------------------------------------
    # GET
    # ------------------------------------------------------

    def do_GET(self):
        if self.path == "/":
            html = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/stats":
            lines = read_lines()
            _, stats = clean(lines)
            self.send_json(200, {
                "total_lines": len(lines),
                "conversations": stats["after"],
                "blank_lines": stats["blank_removed"],
                "duplicates": stats["duplicates_removed"],
                "malformed": stats["malformed"],
            })
            return

        self.send_error(404)

    # ------------------------------------------------------
    # POST
    # ------------------------------------------------------

    def do_POST(self):
        try:
            if self.path == "/add":
                self.handle_add()
            elif self.path == "/cleanup":
                self.handle_cleanup()
            else:
                self.send_error(404)
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_add(self):
        data = self.read_json_body()
        user = normalize(data.get("user", ""))
        assistant = normalize(data.get("assistant", ""))

        if not user or not assistant:
            self.send_json(400, {
                "error": "Both User and Assistant are required."
            })
            return

        conversation = f"<USER> {user} <ASSISTANT> {assistant} <END>"

        existing = read_lines()
        if dedupe_key(conversation) in {
            dedupe_key(line) for line in existing if line.strip()
        }:
            self.send_json(409, {
                "error": "This conversation is already in the file."
            })
            return

        TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)

        # If the file does not currently end with a newline, appending
        # would glue this conversation onto the previous one.
        needs_newline = (
            TRAINING_FILE.exists()
            and TRAINING_FILE.stat().st_size > 0
            and not TRAINING_FILE.read_text(
                encoding="utf-8"
            ).endswith("\n")
        )

        with TRAINING_FILE.open("a", encoding="utf-8") as file:
            if needs_newline:
                file.write("\n")
            file.write(conversation + "\n")

        self.send_json(200, {
            "success": True,
            "conversations": len([
                line for line in read_lines() if line.strip()
            ]),
        })

    def handle_cleanup(self):
        data = self.read_json_body()
        dry_run = bool(data.get("dry_run", False))

        lines = read_lines()
        if not lines:
            self.send_json(400, {
                "error": "There is nothing in training.txt yet."
            })
            return

        cleaned, stats = clean(lines)
        stats["dry_run"] = dry_run

        if dry_run:
            self.send_json(200, stats)
            return

        # Back up before overwriting, so a bad clean-up is recoverable.
        shutil.copyfile(TRAINING_FILE, BACKUP_FILE)
        write_lines(cleaned)

        stats["backup"] = str(BACKUP_FILE)
        self.send_json(200, stats)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print("Training Data Builder")
    print(f"Training file: {TRAINING_FILE}")
    print(f"Open: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    threading.Timer(
        0.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")
    ).start()
    server = HTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("\nServer stopped.")
