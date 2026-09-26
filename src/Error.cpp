/*
 DYNAMICENGINE3D
 AI-Assisted Soft-Body Physics for Unity3D
 By: Elitmers
*/

#include "Error.h"
#include <string>
#include <sstream>

thread_local ErrorCode g_LastError = ErrorCode::None;
thread_local std::string g_LastStackTrace = "";

#if defined(_WIN32)
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

static bool s_symInitialized = false;

static void EnsureSymInit(HANDLE process)
{
    if (s_symInitialized) return;

    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    s_symInitialized = SymInitialize(process, NULL, TRUE) != FALSE;
}

static void CaptureStackTrace()
{
    std::stringstream ss;
    void* stack[32];

    // Capture up to 32 frames, skipping the current CaptureStackTrace frame.
    USHORT frames = CaptureStackBackTrace(1, 32, stack, NULL);
    HANDLE process = GetCurrentProcess();

    EnsureSymInit(process);

    char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
    PSYMBOL_INFO symbol = (PSYMBOL_INFO)buffer;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;

    IMAGEHLP_LINE64 line;
    line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    DWORD displacement = 0;

    for (USHORT i = 0; i < frames; i++)
    {
        DWORD64 address = (DWORD64)(stack[i]);

        if (SymFromAddr(process, address, 0, symbol))
        {
            ss << "  at " << symbol->Name;

            if (SymGetLineFromAddr64(process, address, &displacement, &line))
            {
                ss << " (" << line.FileName << ":" << line.LineNumber << ")";
            }
            ss << " [0x" << std::hex << symbol->Address << "]\n";
        }
        else
        {
            // Symbol lookup failed � usually a missing/mismatched PDB for
            // whatever copy of this module is actually loaded.
            ss << "  at 0x" << std::hex << address << "\n";
        }
    }

    g_LastStackTrace = ss.str();
}

static void ShutdownSym()
{
    if (!s_symInitialized) return;
    SymCleanup(GetCurrentProcess());
    s_symInitialized = false;
}

#elif defined(__ANDROID__) || defined(__linux__)

#include <unwind.h>
#include <dlfcn.h>
#include <cstdint>

// Android replacement for the DbgHelp path above.
//
// _Unwind_Backtrace is used rather than backtrace()/<execinfo.h> because the
// latter is not reliably declared before API 33. dladdr() returns whatever the
// dynamic symbol table holds, so release builds (which strip local symbols)
// resolve the exported API surface but usually not internal statics — traces
// are coarser than the Windows ones.

struct UnwindState
{
    void** frames;
    int    max;
    int    count;
    int    skip;
};

static _Unwind_Reason_Code UnwindCallback(struct _Unwind_Context* ctx, void* arg)
{
    UnwindState* st = static_cast<UnwindState*>(arg);

    // The first frame is this capture helper itself; skip it so the trace
    // starts at the caller, matching CaptureStackBackTrace(1, ...) on Windows.
    if (st->skip > 0)
    {
        --st->skip;
        return _URC_NO_REASON;
    }

    if (st->count >= st->max) return _URC_END_OF_STACK;

    const uintptr_t ip = _Unwind_GetIP(ctx);
    if (ip != 0)
        st->frames[st->count++] = reinterpret_cast<void*>(ip);

    return _URC_NO_REASON;
}

static void CaptureStackTrace()
{
    void* frames[32];
    UnwindState st{ frames, 32, 0, 1 };
    _Unwind_Backtrace(UnwindCallback, &st);

    std::stringstream ss;
    for (int i = 0; i < st.count; ++i)
    {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(frames[i]);
        Dl_info info;

        if (dladdr(frames[i], &info) != 0)
        {
            if (info.dli_sname)
                ss << "  at " << info.dli_sname;
            else if (info.dli_fname)
                ss << "  at " << info.dli_fname;
            else
                ss << "  at <unknown>";
        }
        else
        {
            ss << "  at <unknown>";
        }

        ss << " [0x" << std::hex << addr << "]\n";
    }

    g_LastStackTrace = ss.str();
}

// No symbol-handler resources to release on this platform.
static void ShutdownSym() {}

#else // !_WIN32 && !__ANDROID__ && !__linux__

static void CaptureStackTrace()
{
    g_LastStackTrace = "(stack traces are only implemented on Windows)";
}

static void ShutdownSym() {}

#endif

void Error_SetError(ErrorCode code)
{
    g_LastError = code;
    if (code != ErrorCode::None)
    {
        CaptureStackTrace();
    }
    else
    {
        g_LastStackTrace.clear();
    }
}

EXPORT int Error_GetLastAndClear()
{
    int temp = static_cast<int>(g_LastError);
    g_LastError = ErrorCode::None;
    return temp;
}

EXPORT const char* Error_GetLastStackTrace()
{
    return g_LastStackTrace.c_str();
}

EXPORT void Error_Clear()
{
    g_LastError = ErrorCode::None;
    g_LastStackTrace.clear();
}

EXPORT void Error_Shutdown()
{
    ShutdownSym();
}