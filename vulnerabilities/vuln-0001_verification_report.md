# vuln-0001 Verification Report

**Verification Status**: VALID

**Analysis**
The vulnerability is confirmed to exist in the current codebase. The function `RemoveIfExist` in `src/io/file_io.cc` directly concatenates the user-controlled `path` parameter into a shell command executed via `system()`. This matches the reported vulnerable code exactly:

- File: `src/io/file_io.cc`
- Function: `bool RemoveIfExist(const std::string& path)`
- Vulnerable line (17): `system((\"rm -f \" + path).c_str());`

The function is declared in `src/io/file_io.h`, confirming it is exported and potentially callable from other parts of the codebase. No sanitization or escaping is performed on `path`, allowing command injection via payloads like `\"; touch /tmp/poc\"`. The PoC provided in the report would succeed as-is.

**Evidence**
- `src/io/file_io.cc:17`:
  ```
  17→        system((\"rm -f \" + path).c_str());
  ```
- Full function context (`src/io/file_io.cc` lines 14-20):
  ```
  14→bool RemoveIfExist(const std::string& path) {
  15→    if (PathExists(path)) {
  16→        // LOG(INFO) << \"remove \" << path;
  17→        system((\"rm -f \" + path).c_str());
  18→        return true;
  19→    }
  20→    return false;
  21→}
  ```
- Header declaration (`src/io/file_io.h` lines 20-25):
  ```
  20→/**
  21→ * 若文件存在，则删除之
  22→ * @param path
  23→ * @return
  24→ */
  25→bool RemoveIfExist(const std::string& path);
  ```

**Recommendation**
Proceed with the fix. Implement the suggested remediation using `std::filesystem::remove` to eliminate the shell command injection risk. Test thoroughly after patching, including compilation with C++17 and verification that injection payloads no longer execute arbitrary commands.
