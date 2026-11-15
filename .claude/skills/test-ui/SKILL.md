---
name: test-ui
description: Test UI changes to the backgammon board by launching the app, taking a screenshot to inspect it visually, then closing the app. Use when the user asks to check, verify, or test how a UI change looks.
tools: Bash, Read
---

# Test UI

Launch the backgammon app, capture a screenshot to inspect visually, then shut it down.

## Steps

### 1. Build and launch the app in the background

```bash
make build && .venv/bin/python main.py &
APP_PID=$!
echo $APP_PID
```

Store the PID — you will need it to close the app.

### 2. Wait for the window to appear

```bash
sleep 3
```

Pygame needs a moment to initialise and render the first frame.

### 3. Capture a screenshot

On macOS use `screencapture`. Capture the whole screen (the pygame window will be the only non-system window):

```bash
screencapture -x /tmp/backgammon_ui_test.png
```

### 4. Inspect the screenshot

Use the Read tool to view the image:

```
Read /tmp/backgammon_ui_test.png
```

Describe what you see. Check the specific aspect the user asked about (layout, colours, positioning, text, etc.).

### 5. Close the app

```bash
kill $APP_PID 2>/dev/null; wait $APP_PID 2>/dev/null; true
```

### 6. Report findings

Summarise what the screenshot shows and whether the UI change looks correct. If something is wrong, describe what needs adjusting.

## Reference

The file `images/board_example.png` shows what a correct backgammon board should roughly look like. Read it as a visual reference when evaluating the screenshot.

## Notes

- The `screencapture -x` flag suppresses the shutter sound.
- If the window has not appeared after 3 seconds (e.g. on a slow machine), increase the sleep duration.
- If multiple windows are open, use `screencapture -l <windowid>` with the window ID obtained from `osascript -e 'tell app "System Events" to get id of first window of first process whose name contains "python"'`.
- Always kill the process in step 5 even if an earlier step fails, to avoid leaving a dangling pygame window.