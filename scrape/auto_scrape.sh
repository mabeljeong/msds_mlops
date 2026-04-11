#!/bin/bash
# auto_scrape.sh — run scrape_apartments_sf.py in timed batches automatically.
#
# Usage:
#   bash auto_scrape.sh                   # defaults below
#   START=5 SESSIONS=8 bash auto_scrape.sh  # override via env vars
#
# Edit the variables below to match where you left off.

START=${START:-11}       # first page to scrape (change to where you left off)
BATCH=${BATCH:-2}       # pages per session (keep low to avoid blocks)
SESSIONS=${SESSIONS:-10}  # total number of sessions to run
WAIT=${WAIT:-2400}      # seconds between sessions (default 40 min)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=== auto_scrape.sh ==="
echo "Start page : $START"
echo "Pages/session: $BATCH"
echo "Sessions   : $SESSIONS"
echo "Wait       : ${WAIT}s (~$((WAIT / 60)) min) between sessions"
echo "Total pages: $((SESSIONS * BATCH)) (pages $START – $((START + SESSIONS * BATCH - 1)))"
echo ""

for i in $(seq 0 $((SESSIONS - 1))); do
    PAGE=$((START + i * BATCH))
    echo "──────────────────────────────────────────"
    echo "Session $((i + 1)) / $SESSIONS  |  pages $PAGE – $((PAGE + BATCH - 1))  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "──────────────────────────────────────────"

    python scrape_apartments_sf.py --append --start-page "$PAGE" --max-pages "$BATCH" --chrome

    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "Scraper exited with code $EXIT — stopping automation."
        exit $EXIT
    fi

    if [ $i -lt $((SESSIONS - 1)) ]; then
        echo ""
        echo "Waiting ${WAIT}s (~$((WAIT / 60)) min) before next session..."
        sleep "$WAIT"
    fi
done

echo ""
echo "=== All $SESSIONS sessions complete. ==="
