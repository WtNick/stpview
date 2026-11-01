//! A Windows shell thumbnail provider for STEP and STL files: Explorer draws
//! the part instead of a generic icon.
//!
//! An in-process COM server, loaded into a shell surrogate (`dllhost.exe`),
//! implementing `IThumbnailProvider` over `IInitializeWithStream`.
//! `stpview --register` writes the HKCU keys that name it, and
//! `stpview --unregister` takes them away again.
//!
//! The picture itself is [`stpview::thumb::render`], which `stpview --thumb`
//! also runs, so failures can be reproduced from a command line; inside the
//! shell a failure just shows the generic icon. `STPVIEW_THUMB_LOG` logs them.

#![cfg(windows)]
// The COM entry points are exported by name, and MSVC warns (LNK4104) about
// entry points in the import library. Nothing links against this DLL.
#![allow(linker_messages)]

mod bitmap;

use std::panic::AssertUnwindSafe;
use std::sync::Mutex;

use windows::Win32::Foundation::{
    CLASS_E_CLASSNOTAVAILABLE, CLASS_E_NOAGGREGATION, E_FAIL, E_INVALIDARG, E_POINTER, S_FALSE,
};
use windows::Win32::Graphics::Gdi::HBITMAP;
use windows::Win32::System::Com::{IClassFactory, IClassFactory_Impl, IStream};
use windows::Win32::UI::Shell::PropertiesSystem::{
    IInitializeWithStream, IInitializeWithStream_Impl,
};
use windows::Win32::UI::Shell::{IThumbnailProvider, IThumbnailProvider_Impl, WTSAT_RGB, WTS_ALPHATYPE};
use windows::core::{BOOL, ComObject, GUID, HRESULT, Interface, Ref, implement};

/// The provider's class. Fixed: registry keys are named after it, and changing
/// it would orphan existing registrations.
pub const CLSID_THUMBNAIL: GUID = GUID::from_u128(0x7f9a2c41_3b8e_4d6a_9e51_2c7d8b4f6a03);

/// Where a thumbnail provider hangs off a file type: the shell's own interface
/// identifier for `IThumbnailProvider`, used as the name of a `ShellEx` subkey.
/// Fixed by Windows, not by this crate.
pub const SHELLEX_THUMBNAIL_PROVIDER: &str = "{e357fccd-a995-4576-b01f-234630154e96}";

/// The largest file thumbnailed, in megabytes, unless `STPVIEW_THUMB_MAX_MB`
/// says otherwise.
///
/// The surrogate is shared with every other shell extension, so huge
/// assemblies are not worth their memory; at thumbnail size they are a smudge
/// anyway, and single parts are small.
const MAX_MB: u64 = 64;

/// The largest thumbnail that will be rendered, whatever is asked for.
///
/// Returning a bitmap smaller than `cx` is allowed, and render cost grows with
/// the square of this. Explorer's extra-large view asks for 256.
const MAX_CX: u32 = 1024;

fn max_bytes() -> usize {
    let mb = std::env::var("STPVIEW_THUMB_MAX_MB")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(MAX_MB);
    (mb.saturating_mul(1024 * 1024)) as usize
}

/// Appends a line to the file named by `STPVIEW_THUMB_LOG`, if it is set.
/// Errors writing the log are ignored, since this runs in a shared process.
///
/// The surrogate is started by COM and does not inherit a console's
/// environment: set the variable with `setx`. It is read per call, but a
/// running surrogate sees it only once restarted.
fn log(line: &str) {
    let Ok(path) = std::env::var("STPVIEW_THUMB_LOG") else {
        return;
    };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "[{}] {line}", std::process::id());
    }
}

/// Runs a COM method body with unwinding stopped at the boundary.
///
/// A panic unwinding into the vtable thunk is undefined behaviour, and would
/// take down a `dllhost.exe` shared with other shell extensions.
/// `AssertUnwindSafe` holds because no state outlives a failed call.
fn guard<T>(what: &str, body: impl FnOnce() -> windows::core::Result<T>) -> windows::core::Result<T> {
    match std::panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => {
            log(&format!("{what} failed: {e}"));
            Err(e)
        }
        Err(_) => {
            log(&format!("{what} panicked"));
            Err(E_FAIL.into())
        }
    }
}

/// One request: the shell creates the object, initialises it with the file, and
/// asks for one bitmap.
#[implement(IThumbnailProvider, IInitializeWithStream)]
#[derive(Default)]
struct Thumbnail {
    /// The file's bytes, once `Initialize` has read the stream.
    src: Mutex<Vec<u8>>,
}

impl IInitializeWithStream_Impl for Thumbnail_Impl {
    fn Initialize(&self, stream: Ref<IStream>, _mode: u32) -> windows::core::Result<()> {
        guard("Initialize", || {
            let stream = stream.ok()?;
            let bytes = read_capped(stream, max_bytes())?;
            *self.src.lock().map_err(|_| E_FAIL)? = bytes;
            Ok(())
        })
    }
}

/// Reads the whole stream, refusing anything over `cap`.
///
/// The stream's `Stat` is not consulted: the cap applies to bytes actually
/// read, which works for streams of unknown or misreported length.
fn read_capped(stream: &IStream, cap: usize) -> windows::core::Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    let mut chunk = vec![0u8; 256 * 1024];
    loop {
        let mut got = 0u32;
        // `Read` returns S_FALSE (a success code) at the end; the byte count
        // says stop.
        unsafe {
            stream
                .Read(chunk.as_mut_ptr().cast(), chunk.len() as u32, Some(&mut got))
                .ok()?
        };
        if got == 0 {
            return Ok(out);
        }
        if out.len() + got as usize > cap {
            log(&format!("refused: over {} bytes", cap));
            return Err(E_FAIL.into());
        }
        out.extend_from_slice(&chunk[..got as usize]);
    }
}

impl IThumbnailProvider_Impl for Thumbnail_Impl {
    fn GetThumbnail(
        &self,
        cx: u32,
        phbmp: *mut HBITMAP,
        pdwalpha: *mut WTS_ALPHATYPE,
    ) -> windows::core::Result<()> {
        guard("GetThumbnail", || {
            if phbmp.is_null() || pdwalpha.is_null() {
                return Err(E_POINTER.into());
            }
            // Cleared first: a caller is entitled to read these back on
            // failure, and a stale pointer would be deleted as a bitmap.
            unsafe {
                *phbmp = HBITMAP::default();
                *pdwalpha = WTSAT_RGB;
            }
            if cx == 0 {
                return Err(E_INVALIDARG.into());
            }
            let cx = cx.min(MAX_CX);

            let src = self.src.lock().map_err(|_| E_FAIL)?;
            if src.is_empty() {
                // Not initialised, or initialised with nothing.
                return Err(E_FAIL.into());
            }
            let rgba = stpview::thumb::render(&src, cx, "thumbnail").map_err(|e| {
                log(&format!("render failed: {e}"));
                windows::core::Error::new(E_FAIL, e)
            })?;
            drop(src);

            let bgra = bitmap::bgra_from_rgba(&rgba);
            let bmp = unsafe { bitmap::hbitmap(cx as i32, cx as i32, &bgra)? };
            // The shell owns the bitmap from here and deletes it itself.
            unsafe { *phbmp = bmp };
            Ok(())
        })
    }
}

/// The class object the shell asks for before it asks for an instance.
#[implement(IClassFactory)]
struct Factory;

impl IClassFactory_Impl for Factory_Impl {
    fn CreateInstance(
        &self,
        punkouter: Ref<windows::core::IUnknown>,
        riid: *const GUID,
        ppvobject: *mut *mut core::ffi::c_void,
    ) -> windows::core::Result<()> {
        guard("CreateInstance", || {
            if ppvobject.is_null() {
                return Err(E_POINTER.into());
            }
            unsafe { *ppvobject = std::ptr::null_mut() };
            if !punkouter.is_null() {
                return Err(CLASS_E_NOAGGREGATION.into());
            }
            let obj = ComObject::new(Thumbnail::default());
            unsafe { obj.to_interface::<windows::core::IUnknown>().query(riid, ppvobject).ok() }
        })
    }

    fn LockServer(&self, _flock: BOOL) -> windows::core::Result<()> {
        // Nothing to lock: see `DllCanUnloadNow`, which never lets go.
        Ok(())
    }
}

/// The shell's entry point into the DLL.
#[unsafe(no_mangle)]
unsafe extern "system" fn DllGetClassObject(
    rclsid: *const GUID,
    riid: *const GUID,
    ppv: *mut *mut core::ffi::c_void,
) -> HRESULT {
    unsafe {
        if ppv.is_null() {
            return E_POINTER;
        }
        *ppv = std::ptr::null_mut();
        if rclsid.is_null() || riid.is_null() {
            return E_INVALIDARG;
        }
        if *rclsid != CLSID_THUMBNAIL {
            return CLASS_E_CLASSNOTAVAILABLE;
        }
        ComObject::new(Factory)
            .to_interface::<IClassFactory>()
            .query(riid, ppv)
    }
}

/// Always S_FALSE: this DLL is never unloaded from a host that has used it.
///
/// Behind `stpview::thumb` sit a live wgpu device with driver threads and
/// rayon's global pool, neither of which survives `FreeLibrary`. The surrogate
/// process retiring when idle reclaims everything.
#[unsafe(no_mangle)]
extern "system" fn DllCanUnloadNow() -> HRESULT {
    S_FALSE
}

/// The CLSID in the brace-and-hyphen form a registry key is named with.
///
/// Derived from [`CLSID_THUMBNAIL`], so registry keys always name the class
/// the DLL serves.
pub fn clsid_string() -> String {
    let g = CLSID_THUMBNAIL;
    format!(
        "{{{:08X}-{:04X}-{:04X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}}}",
        g.data1,
        g.data2,
        g.data3,
        g.data4[0],
        g.data4[1],
        g.data4[2],
        g.data4[3],
        g.data4[4],
        g.data4[5],
        g.data4[6],
        g.data4[7],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clsid_string_is_the_clsid() {
        // Registry keys use this string; the DLL answers for the GUID.
        assert_eq!(clsid_string(), "{7F9A2C41-3B8E-4D6A-9E51-2C7D8B4F6A03}");
    }

    #[test]
    fn the_cap_can_be_overridden_but_defaults() {
        // Only the default: the environment is shared with other tests.
        assert_eq!(max_bytes(), (MAX_MB * 1024 * 1024) as usize);
    }
}
