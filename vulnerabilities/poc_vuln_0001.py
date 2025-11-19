# PoC to demonstrate and verify fix for vuln-0001: Command Injection in RemoveIfExist
# Run this script to:
# 1. Test vulnerable version (creates marker file via injection)
# 2. Test fixed version (no injection)

import os
import subprocess
import tempfile
import shutil
import sys

# Use temp dir for marker
temp_dir = tempfile.gettempdir()
marker_file = os.path.join(temp_dir, 'rce_proof_vuln0001.txt')

def cleanup():
    if os.path.exists(marker_file):
        os.remove(marker_file)
    for ext in ['.cc', '.exe', '.out']:
        for name in ['vuln_test', 'fixed_test']:
            test_file = name + ext
            if os.path.exists(test_file):
                os.remove(test_file)

def compile_and_run(cc_content, exe_name, expected_vuln):
    with open(f'{exe_name}.cc', 'w') as f:
        f.write(cc_content)

    compile_cmd = ['g++', f'{exe_name}.cc', '-o', exe_name, '-std=c++17']
    if sys.platform == 'win32':
        compile_cmd[4] = exe_name + '.exe'

    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'Compile failed for {exe_name}: {result.stderr}')
        return False

    run_cmd = ['./' + exe_name] if sys.platform != 'win32' else [exe_name + '.exe']
    run_result = subprocess.run(run_cmd, capture_output=True, text=True)
    print(f'{exe_name} output: {run_result.stdout}')

    injected = os.path.exists(marker_file)
    status = 'INJECTED!' if injected == expected_vuln else 'SAFE (no injection)'
    print(f'{exe_name}: {status}')
    return injected

def main():
    cleanup()

    # Vulnerable code snippet
    malicious_path = r'foo; echo VULN_RCE_PROVEN > ' + marker_file.replace('\\', '/') + r' 2>/dev/null || (echo VULN_RCE_PROVEN > ' + marker_file.replace('\\', '/') + r')'
    if sys.platform == 'win32':
        malicious_path = r'foo & echo VULN_RCE_PROVEN > ' + marker_file.replace('\\', '/') + r' 2>nul'

    vuln_cc = f'''#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>

int main() {{
    std::string path = R"({malicious_path})";
    std::cout << "Testing vulnerable RemoveIfExist logic..." << std::endl;
    std::cout << "Command: rm -f " << path << std::endl;
    system(("rm -f " + path).c_str());
    return 0;
}}'''

    print('=== Testing VULNERABLE version ===')
    vuln_injected = compile_and_run(vuln_cc, 'vuln_test', True)

    # Fixed code snippet
    fixed_cc = f'''#include <iostream>
#include <string>
#include <filesystem>

bool PathExists(const std::string& file_path) {{
    return std::filesystem::exists(file_path);
}}

int main() {{
    std::string path = R"({malicious_path})";
    std::cout << "Testing FIXED RemoveIfExist logic..." << std::endl;
    if (PathExists(path)) {{
        try {{
            std::filesystem::remove(std::filesystem::path(path));
        }} catch (...) {{}}
    }}
    return 0;
}}'''

    print('\\n=== Testing FIXED version ===')
    fixed_injected = compile_and_run(fixed_cc, 'fixed_test', False)

    if vuln_injected and not fixed_injected:
        print('\\n✅ PoC SUCCESS: Vulnerability demonstrated and fix verified!')
    else:
        print('\\n❌ PoC FAILED: Check compilation/runtime issues.')

    cleanup()

if __name__ == '__main__':
    main()
