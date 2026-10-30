"""Check Streamlit container health and its HTTP application shell."""

import argparse
import time
from urllib.error import URLError
from urllib.request import urlopen


def check(url, attempts):
    """Allow startup retries; domain/AppTest tests verify the rendered app logic."""
    for attempt in range(attempts):
        try:
            with urlopen(url + "/_stcore/health", timeout=15) as response:
                if response.status != 200:
                    raise ValueError("Streamlit not healthy")
            with urlopen(url, timeout=15) as response:
                if b"streamlit" not in response.read().lower():
                    raise ValueError("Missing Streamlit HTTP shell")
            print("Streamlit health and HTTP shell passed:", url)
            return
        except (OSError, URLError, TimeoutError, ValueError) as exc:
            if attempt + 1 == attempts:
                raise SystemExit(str(exc)) from exc
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--attempts", type=int, default=6)
    args = parser.parse_args()
    check(args.url.rstrip("/"), args.attempts)
