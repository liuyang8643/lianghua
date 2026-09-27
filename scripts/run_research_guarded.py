"""Windows research launcher; a private kill-on-close Job owns the entire child tree.

No Torch/runtime imports. Game detection fails closed. A stop discards in-flight work;
only previously verified checkpoints may be resumed. Never auto-restarts a stopped run.
This is a bounded polling safeguard, not a guarantee of zero scheduling/GPU latency.
"""
from __future__ import annotations

import argparse
import ctypes as ct
from ctypes import wintypes as wt
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import time
import uuid

GAME_IMAGES = frozenset({'leagueclient.exe', 'leagueclientux.exe', 'league of legends.exe'})
POLL_SECONDS = 0.25
JOB_ENV = 'WBR_RESEARCH_GUARD_JOB'


class ProcessEntry(ct.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD), ('th32ProcessID', wt.DWORD),
                ('th32DefaultHeapID', ct.c_size_t), ('th32ModuleID', wt.DWORD),
                ('cntThreads', wt.DWORD), ('th32ParentProcessID', wt.DWORD),
                ('pcPriClassBase', wt.LONG), ('dwFlags', wt.DWORD), ('szExeFile', wt.WCHAR * 260)]


class BasicLimits(ct.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ct.c_int64), ('PerJobUserTimeLimit', ct.c_int64),
                ('LimitFlags', wt.DWORD), ('MinimumWorkingSetSize', ct.c_size_t),
                ('MaximumWorkingSetSize', ct.c_size_t), ('ActiveProcessLimit', wt.DWORD),
                ('Affinity', ct.c_size_t), ('PriorityClass', wt.DWORD), ('SchedulingClass', wt.DWORD)]


class IoCounters(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount',
        'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class ExtendedLimits(ct.Structure):
    _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IoCounters),
                ('ProcessMemoryLimit', ct.c_size_t), ('JobMemoryLimit', ct.c_size_t),
                ('PeakProcessMemoryUsed', ct.c_size_t), ('PeakJobMemoryUsed', ct.c_size_t)]


class StartupInfo(ct.Structure):
    _fields_ = [('cb', wt.DWORD), ('lpReserved', wt.LPWSTR), ('lpDesktop', wt.LPWSTR),
        ('lpTitle', wt.LPWSTR), *[(name, wt.DWORD) for name in
        ('dwX', 'dwY', 'dwXSize', 'dwYSize', 'dwXCountChars', 'dwYCountChars', 'dwFillAttribute', 'dwFlags')],
        ('wShowWindow', wt.WORD), ('cbReserved2', wt.WORD), ('lpReserved2', ct.c_void_p),
        ('hStdInput', wt.HANDLE), ('hStdOutput', wt.HANDLE), ('hStdError', wt.HANDLE)]


class StartupInfoEx(ct.Structure):
    _fields_ = [('StartupInfo', StartupInfo), ('lpAttributeList', ct.c_void_p)]


class ProcessInfo(ct.Structure):
    _fields_ = [('hProcess', wt.HANDLE), ('hThread', wt.HANDLE),
                ('dwProcessId', wt.DWORD), ('dwThreadId', wt.DWORD)]


class JobAccounting(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in ('TotalUserTime', 'TotalKernelTime',
        'ThisPeriodTotalUserTime', 'ThisPeriodTotalKernelTime')] + [(name, wt.DWORD) for name in
        ('TotalPageFaultCount', 'TotalProcesses', 'ActiveProcesses', 'TotalTerminatedProcesses')]


def win_api():
    if os.name != 'nt':
        raise RuntimeError('This guard requires Windows Job Objects')
    dll = ct.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateToolhelp32Snapshot': ([wt.DWORD, wt.DWORD], wt.HANDLE),
        'Process32FirstW': ([wt.HANDLE, ct.POINTER(ProcessEntry)], wt.BOOL),
        'Process32NextW': ([wt.HANDLE, ct.POINTER(ProcessEntry)], wt.BOOL),
        'CreateJobObjectW': ([ct.c_void_p, wt.LPCWSTR], wt.HANDLE),
        'OpenJobObjectW': ([wt.DWORD, wt.BOOL, wt.LPCWSTR], wt.HANDLE),
        'IsProcessInJob': ([wt.HANDLE, wt.HANDLE, ct.POINTER(wt.BOOL)], wt.BOOL),
        'SetInformationJobObject': ([wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD], wt.BOOL),
        'QueryInformationJobObject': ([wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p], wt.BOOL),
        'TerminateJobObject': ([wt.HANDLE, wt.UINT], wt.BOOL),
        'CloseHandle': ([wt.HANDLE], wt.BOOL),
        'InitializeProcThreadAttributeList': ([ct.c_void_p, wt.DWORD, wt.DWORD, ct.POINTER(ct.c_size_t)], wt.BOOL),
        'UpdateProcThreadAttribute': ([ct.c_void_p, wt.DWORD, ct.c_size_t, ct.c_void_p,
            ct.c_size_t, ct.c_void_p, ct.c_void_p], wt.BOOL),
        'DeleteProcThreadAttributeList': ([ct.c_void_p], None),
        'CreateProcessW': ([wt.LPCWSTR, wt.LPWSTR, ct.c_void_p, ct.c_void_p, wt.BOOL,
            wt.DWORD, ct.c_void_p, wt.LPCWSTR, ct.POINTER(StartupInfoEx), ct.POINTER(ProcessInfo)], wt.BOOL),
        'ResumeThread': ([wt.HANDLE], wt.DWORD),
        'WaitForSingleObject': ([wt.HANDLE, wt.DWORD], wt.DWORD),
        'GetExitCodeProcess': ([wt.HANDLE, ct.POINTER(wt.DWORD)], wt.BOOL),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(dll, name)
        fn.argtypes, fn.restype = args, result
    return dll


def checked(ok):
    if not ok:
        raise ct.WinError(ct.get_last_error())
    return ok


def game_processes():
    api = win_api()
    snapshot = api.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ct.c_void_p(-1).value:
        raise ct.WinError(ct.get_last_error())
    found = []
    try:
        entry = ProcessEntry()
        entry.dwSize = ct.sizeof(entry)
        more = api.Process32FirstW(snapshot, ct.byref(entry))
        while more:
            if entry.szExeFile.casefold() in GAME_IMAGES:
                found.append({'pid': entry.th32ProcessID, 'image': entry.szExeFile})
            more = api.Process32NextW(snapshot, ct.byref(entry))
        if ct.get_last_error() != 18:  # ERROR_NO_MORE_FILES is the only expected end.
            raise ct.WinError(ct.get_last_error())
    finally:
        checked(api.CloseHandle(snapshot))
    return found


def require_guarded_job():
    """Reject a direct worker invocation, including one inside an unrelated outer Job."""
    name = os.environ.get(JOB_ENV)
    if not name or not name.startswith('Local\\WBRResearchGuard-'):
        raise RuntimeError('Worker must be launched through the research game guard')
    api = win_api()
    job = checked(api.OpenJobObjectW(4, False, name))
    try:
        member = wt.BOOL()
        checked(api.IsProcessInJob(wt.HANDLE(-1), job, ct.byref(member)))
        limits = ExtendedLimits()
        checked(api.QueryInformationJobObject(job, 9, ct.byref(limits), ct.sizeof(limits), None))
        flags = limits.BasicLimitInformation.LimitFlags
        if not member.value or not flags & 0x2000 or flags & (0x800 | 0x1000):
            raise RuntimeError('Worker is not owned by a non-breakaway kill-on-close research Job')
    finally:
        checked(api.CloseHandle(job))


def create_owned_process(api, job, command, cwd, stdout_path, stderr_path, job_name):
    """Windows 10+ JOB_LIST assigns ownership atomically at process creation."""
    size = ct.c_size_t()
    api.InitializeProcThreadAttributeList(None, 2, 0, ct.byref(size))
    if ct.get_last_error() != 122:  # expected size query: ERROR_INSUFFICIENT_BUFFER
        raise ct.WinError(ct.get_last_error())
    buffer = ct.create_string_buffer(size.value)
    checked(api.InitializeProcThreadAttributeList(buffer, 2, 0, ct.byref(size)))
    try:
        jobs = (wt.HANDLE * 1)(job)
        checked(api.UpdateProcThreadAttribute(buffer, 0, 0x2000D, jobs, ct.sizeof(jobs), None, None))
        with open(os.devnull, 'rb') as source, Path(stdout_path).open('ab') as out, Path(stderr_path).open('ab') as err:
            handles = (wt.HANDLE * 3)(*[msvcrt.get_osfhandle(stream.fileno()) for stream in (source, out, err)])
            for handle in handles:
                os.set_handle_inheritable(handle, True)
            checked(api.UpdateProcThreadAttribute(buffer, 0, 0x20002, handles, ct.sizeof(handles), None, None))
            info = StartupInfoEx()
            info.StartupInfo.cb = ct.sizeof(info)
            info.StartupInfo.dwFlags = 0x100  # STARTF_USESTDHANDLES
            info.StartupInfo.hStdInput, info.StartupInfo.hStdOutput, info.StartupInfo.hStdError = handles
            info.lpAttributeList = ct.cast(buffer, ct.c_void_p)
            process = ProcessInfo()
            environment = dict(os.environ)
            environment[JOB_ENV] = job_name
            env_block = ct.create_unicode_buffer('\0'.join(f'{key}={value}' for key, value in
                sorted(environment.items(), key=lambda item: item[0].upper())) + '\0\0')
            flags = 0x4 | 0x80000 | 0x400 | subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            checked(api.CreateProcessW(None, ct.create_unicode_buffer(subprocess.list2cmdline(command)),
                None, None, True, flags, env_block, str(cwd), ct.byref(info), ct.byref(process)))
            return process
    finally:
        api.DeleteProcThreadAttributeList(buffer)


def run_guarded(command, *, cwd, event_path, stdout_path, stderr_path,
                detector=game_processes):
    """Detector injection is for lightweight process tests, not exposed by the CLI."""
    api = win_api()
    def record(event, **data):
        with Path(event_path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event=event, time=time.time(), **data)) + '\n')

    games = detector()
    if games:
        record('blocked_before_launch', games=games)
        return 75
    job_name = 'Local\\WBRResearchGuard-' + uuid.uuid4().hex
    job = checked(api.CreateJobObjectW(None, job_name))
    process = None
    try:
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE; no breakaway.
        checked(api.SetInformationJobObject(job, 9, ct.byref(limits), ct.sizeof(limits)))
        process = create_owned_process(api, job, command, cwd, stdout_path, stderr_path, job_name)
        games = detector()
        if games:
            record('blocked_before_resume', pid=process.dwProcessId, games=games)
            return 75
        if api.ResumeThread(process.hThread) == 0xFFFFFFFF:
            raise ct.WinError(ct.get_last_error())
        record('started', pid=process.dwProcessId, poll_seconds=POLL_SECONDS)
        while True:
            state = api.WaitForSingleObject(process.hProcess, 0)
            if state == 0:
                break
            if state != 258:
                raise ct.WinError(ct.get_last_error())
            games = detector()
            if games:
                checked(api.TerminateJobObject(job, 75))
                record('stopped_for_game', pid=process.dwProcessId, games=games)
                return 75
            time.sleep(POLL_SECONDS)
        exit_code = wt.DWORD()
        checked(api.GetExitCodeProcess(process.hProcess, ct.byref(exit_code)))
        record('child_exit', pid=process.dwProcessId, returncode=exit_code.value)
        return exit_code.value
    finally:
        # Runs on detector/logging errors too. OS also closes the Job if guard crashes.
        try:
            checked(api.TerminateJobObject(job, 75))
            deadline = time.monotonic() + 10
            while True:
                accounting = JobAccounting()
                checked(api.QueryInformationJobObject(job, 1, ct.byref(accounting), ct.sizeof(accounting), None))
                if accounting.ActiveProcesses == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Owned process tree did not terminate within 10 seconds')
                time.sleep(.01)
            if process is not None:
                if api.WaitForSingleObject(process.hProcess, 10_000) != 0:
                    raise RuntimeError('Owned process did not terminate within 10 seconds')
        finally:
            checked(api.CloseHandle(job))
            if process is not None:
                checked(api.CloseHandle(process.hThread))
                checked(api.CloseHandle(process.hProcess))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Supply an executable and arguments after --')
    args.logs.mkdir(parents=True, exist_ok=True)
    return run_guarded(command, cwd=args.cwd, event_path=args.logs/'guard.jsonl',
        stdout_path=args.logs/'stdout.log', stderr_path=args.logs/'stderr.log')


if __name__ == '__main__':
    raise SystemExit(main())
