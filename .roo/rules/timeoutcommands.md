# Command Execution Safety Rules (Windows Command Prompt)

## Purpose
To prevent indefinite blocking, hanging processes, or waiting for user input during command execution in **Windows Command Prompt (`cmd.exe`)** environments.

## General Rule
Any command executed via **Windows Command Prompt** that could hang, wait for input, or run indefinitely **MUST** be executed with an enforced timeout or an equivalent termination mechanism.

## Implementation

### 1. **Apply Timeout**
- All commands must be wrapped with a timeout or external watchdog mechanism.
- Native `cmd.exe` does **not** provide a built-in per-command timeout; therefore, one of the following approaches **must** be used:

**Option A: PowerShell wrapper (preferred)**
```cmd
powershell -Command "Start-Process cmd.exe -ArgumentList '/c <command>' -Wait -Timeout 30"
```

**Option B: PowerShell direct execution**
```cmd
powershell -Command "& { $p = Start-Process '<command>' -PassThru; if (-not $p.WaitForExit(30000)) { $p.Kill() } }"
```

**Option C: Python subprocess**
```python
subprocess.run(cmd, timeout=30, check=False, text=True)
```

### 2. **Default Timeout**
- Default timeout duration: **30 seconds**.
- If a command is expected to take longer, the timeout **must be explicitly increased** (e.g., 120 seconds) and documented.

### 3. **User Input Detection**
- Commands that may prompt for user input must be executed in **non-interactive mode** when available.
- Use command flags such as:
  - `/Y`, `/Q`, `/S`, `/F`, or `/C` where supported.
- Redirect standard input from `NUL` when possible.

**Example:**
```cmd
<command> < NUL
```

**PowerShell equivalent:**
```powershell
$null | <command>
```

### 4. **Failure Handling**
- If a timeout occurs, log:
  > “Command terminated after exceeding timeout limit.”
- Do **not** automatically retry the command unless explicitly instructed.

### 5. **Safe Whitelist**
The following lightweight commands may omit timeouts **only if their arguments are static and known**:
- `dir`
- `cd`
- `echo`
- `type <file>`
- `date /t`
- `time /t`
- `whoami`
- `node.exe`

### 6. **Never Execute Interactively**
Do **not** execute commands that require sustained user interaction, including but not limited to:
- Text editors (`notepad`)
- Pagers or viewers that wait for input
- Interactive shells (`cmd`, `powershell` without `-Command`)
- Debuggers or REPL environments

## Summary
> Always assume Windows commands can hang.  
> Always enforce a timeout using PowerShell or an external mechanism.  
> Always prefer non-interactive execution and explicit termination safeguards.

