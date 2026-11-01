//! Registering the thumbnail provider with Explorer, and taking it back out.
//!
//! Everything goes under `HKEY_CURRENT_USER\Software\Classes`, which needs no
//! administrator: a COM class and a `ShellEx` entry per extension naming it.
//! The CLSID and interface key come from `stpview_shell`, so they always match
//! the class the DLL serves.
//!
//! `ShellEx` is written on the extension rather than a ProgID: Explorer looks
//! there first, and an unclaimed `.step` has only the shell's placeholder
//! `STEP_auto_file` ProgID.

use windows::Win32::Foundation::ERROR_SUCCESS;
use windows::Win32::System::Registry::{
    HKEY, HKEY_CLASSES_ROOT, HKEY_CURRENT_USER, KEY_WRITE, REG_OPTION_NON_VOLATILE, REG_SZ,
    RRF_RT_REG_SZ, RegCloseKey, RegCreateKeyExW, RegDeleteTreeW, RegDeleteValueW, RegGetValueW,
    RegOpenKeyExW, RegSetValueExW,
};
use windows::Win32::UI::Shell::{SHCNE_ASSOCCHANGED, SHCNF_IDLIST, SHChangeNotify};
use windows::core::PCWSTR;

/// The extensions a STEP file is written with. `.stpz` is left out: it is a
/// zipped container, and the provider is handed the bytes as they are.
const EXTENSIONS: [&str; 2] = [".step", ".stp"];

/// Every extension the thumbnail provider draws; it tells STEP from STL by
/// content. The ProgID below is for STEP only.
const THUMBNAILED: [&str; 3] = [".step", ".stp", ".stl"];

const DLL: &str = "stpview_shell.dll";

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Creates `Software\Classes\...` under HKCU and sets one string value in it.
/// `name` of `None` sets the key's default value.
fn set_string(subkey: &str, name: Option<&str>, data: &str) -> Result<(), String> {
    let path = wide(subkey);
    let mut key = HKEY::default();
    let rc = unsafe {
        RegCreateKeyExW(
            HKEY_CURRENT_USER,
            PCWSTR(path.as_ptr()),
            None,
            PCWSTR::null(),
            REG_OPTION_NON_VOLATILE,
            KEY_WRITE,
            None,
            &mut key,
            None,
        )
    };
    if rc != ERROR_SUCCESS {
        return Err(format!("cannot create HKCU\\{subkey}: error {}", rc.0));
    }

    // The byte length must include the terminating NUL.
    let value = wide(data);
    let bytes = unsafe {
        std::slice::from_raw_parts(value.as_ptr() as *const u8, std::mem::size_of_val(&value[..]))
    };
    let named = name.map(wide);
    let rc = unsafe {
        RegSetValueExW(
            key,
            match &named {
                Some(n) => PCWSTR(n.as_ptr()),
                None => PCWSTR::null(),
            },
            None,
            REG_SZ,
            Some(bytes),
        )
    };
    let _ = unsafe { RegCloseKey(key) };
    if rc != ERROR_SUCCESS {
        return Err(format!(
            "cannot set {} of HKCU\\{subkey}: error {}",
            name.unwrap_or("(default)"),
            rc.0
        ));
    }
    Ok(())
}

/// Reads a string value from `HKEY_CLASSES_ROOT`, the merged per-user and
/// per-machine view the shell resolves file types through. Writes go to HKCU.
fn read_class(subkey: &str, name: Option<&str>) -> Option<String> {
    read_string(HKEY_CLASSES_ROOT, subkey, name)
}

fn read_string(root: HKEY, subkey: &str, name: Option<&str>) -> Option<String> {
    let path = wide(subkey);
    let named = name.map(wide);
    let value = match &named {
        Some(n) => PCWSTR(n.as_ptr()),
        None => PCWSTR::null(),
    };

    let mut bytes: u32 = 0;
    let rc = unsafe {
        RegGetValueW(
            root,
            PCWSTR(path.as_ptr()),
            value,
            RRF_RT_REG_SZ,
            None,
            None,
            Some(&mut bytes),
        )
    };
    if rc != ERROR_SUCCESS || bytes == 0 {
        return None;
    }

    let mut buf = vec![0u16; bytes as usize / 2 + 1];
    let mut got = bytes;
    let rc = unsafe {
        RegGetValueW(
            root,
            PCWSTR(path.as_ptr()),
            value,
            RRF_RT_REG_SZ,
            None,
            Some(buf.as_mut_ptr().cast()),
            Some(&mut got),
        )
    };
    if rc != ERROR_SUCCESS {
        return None;
    }
    // The returned length counts the terminating NUL.
    let n = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    Some(String::from_utf16_lossy(&buf[..n]))
}

/// Deletes a key and everything under it. A missing key is not an error, so
/// `--unregister` can clean up a half-finished `--register`.
fn delete_tree(subkey: &str) -> Result<(), String> {
    let path = wide(subkey);
    let rc = unsafe { RegDeleteTreeW(HKEY_CURRENT_USER, PCWSTR(path.as_ptr())) };
    // ERROR_FILE_NOT_FOUND is 2.
    if rc != ERROR_SUCCESS && rc.0 != 2 {
        return Err(format!("cannot delete HKCU\\{subkey}: error {}", rc.0));
    }
    Ok(())
}

/// Deletes one value, leaving the key and its other values alone. A value that
/// was never there is not an error.
fn delete_value(subkey: &str, name: &str) -> Result<(), String> {
    let path = wide(subkey);
    let mut key = HKEY::default();
    let rc = unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            PCWSTR(path.as_ptr()),
            None,
            KEY_WRITE,
            &mut key,
        )
    };
    if rc != ERROR_SUCCESS {
        return Ok(());
    }
    let value = wide(name);
    let rc = unsafe { RegDeleteValueW(key, PCWSTR(value.as_ptr())) };
    let _ = unsafe { RegCloseKey(key) };
    // 2 is ERROR_FILE_NOT_FOUND.
    if rc != ERROR_SUCCESS && rc.0 != 2 {
        return Err(format!("cannot clear {name} of HKCU\\{subkey}: error {}", rc.0));
    }
    Ok(())
}

/// Tells the shell its file associations have changed, so a folder already open
/// picks the provider up without a sign-out.
fn notify_shell() {
    unsafe { SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None) };
}

fn clsid_key() -> String {
    format!("Software\\Classes\\CLSID\\{}", stpview_shell::clsid_string())
}

/// Our own ProgID: a file type with a name, an icon and a way to open it.
const PROGID: &str = "stpview.step";

/// The document icon, as a `DefaultIcon` value.
///
/// A negative index names a resource id (here 2, set by the build script)
/// rather than a position in resource order, which shifts as icons are added.
fn doc_icon(exe: &str) -> String {
    format!("{exe},-2")
}

/// Every ProgID that could decide the icon for this extension.
///
/// `HKCR\<ext>` alone is not enough: a user choice under `FileExts\<ext>`
/// (`UserChoice` or `UserChoiceLatest`, e.g. `Applications\stpview.exe` after
/// Open With) overrides it. All three are collected, and whichever names this
/// executable gets the icon, since there is no telling which the shell uses.
fn icon_progids(ext: &str) -> Vec<String> {
    let exts = format!("Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\{ext}");
    let mut out = Vec::new();
    for (sub, name) in [
        (format!("{exts}\\UserChoice"), Some("ProgId")),
        (format!("{exts}\\UserChoiceLatest\\ProgId"), Some("ProgId")),
    ] {
        if let Some(p) = read_string(HKEY_CURRENT_USER, &sub, name) {
            out.push(p);
        }
    }
    if let Some(p) = read_class(ext, None) {
        out.push(p);
    }
    out.retain(|p| !p.is_empty() && p != PROGID);
    out.sort();
    out.dedup();
    out
}

/// Whether a ProgID's open command runs this executable.
///
/// Guards writing an icon into a ProgID we did not create: a shell-invented
/// one like `STEP_auto_file` may need it, but another application's file type
/// must never be redecorated.
fn progid_opens_with(progid: &str, exe: &str) -> bool {
    read_class(&format!("{progid}\\shell\\open\\command"), None)
        .map(|c| c.to_ascii_lowercase().contains(&exe.to_ascii_lowercase()))
        .unwrap_or(false)
}

fn shellex_key(ext: &str) -> String {
    format!(
        "Software\\Classes\\{ext}\\ShellEx\\{}",
        stpview_shell::SHELLEX_THUMBNAIL_PROVIDER
    )
}

/// The DLL that goes with this executable, as the string the registry will hold
/// it as.
///
/// Cargo puts the two in the same directory; pointing anywhere else risks
/// loading a stale handler. The path must convert to a string losslessly.
fn dll_path() -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate this executable: {e}"))?;
    let dll = exe.parent().ok_or("this executable has no directory")?.join(DLL);
    if !dll.is_file() {
        return Err(format!(
            "{} is not next to the executable; build the workspace with `cargo build --release`",
            dll.display()
        ));
    }
    registry_string(&dll)
}

/// This executable, as the registry will hold it: it carries both icons, and
/// the ProgID's open command runs it.
fn exe_path() -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate this executable: {e}"))?;
    registry_string(&exe)
}

fn registry_string(p: &std::path::Path) -> Result<String, String> {
    p.to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{} cannot be written to the registry", p.display()))
}

pub fn register() -> Result<(), String> {
    let dll = dll_path()?;
    let clsid = stpview_shell::clsid_string();
    let key = clsid_key();

    set_string(&key, None, "stpview STEP and STL thumbnail provider")?;
    set_string(&format!("{key}\\InprocServer32"), None, &dll)?;
    // Apartment, as documented for thumbnail providers: the shell serialises
    // calls, so the handler need not be thread-safe.
    set_string(&format!("{key}\\InprocServer32"), Some("ThreadingModel"), "Apartment")?;

    // Leave another per-user provider in place (slicers often draw STL):
    // overwriting it would lose it on --unregister. A per-machine one is only
    // shadowed.
    let mut taken = Vec::new();
    for ext in THUMBNAILED {
        match read_string(HKEY_CURRENT_USER, &shellex_key(ext), None) {
            Some(other) if !other.eq_ignore_ascii_case(&clsid) => taken.push((ext, other)),
            _ => set_string(&shellex_key(ext), None, &clsid)?,
        }
    }

    // A file type of our own: a name, the document icon, and a way to open it.
    let exe = exe_path()?;
    let progid = format!("Software\\Classes\\{PROGID}");
    set_string(&progid, None, "STEP 3D model")?;
    set_string(&format!("{progid}\\DefaultIcon"), None, &doc_icon(&exe))?;
    set_string(
        &format!("{progid}\\shell\\open\\command"),
        None,
        &format!("\"{exe}\" \"%1\""),
    )?;

    // Registering a ProgID does not make it the default (only the user can),
    // so the icon also goes on whatever ProgID the files resolve to, if that
    // opens with this executable.
    let mut decorated = Vec::new();
    for ext in EXTENSIONS {
        set_string(
            &format!("Software\\Classes\\{ext}\\OpenWithProgids"),
            Some(PROGID),
            "",
        )?;
        for p in icon_progids(ext) {
            if progid_opens_with(&p, &exe) {
                set_string(
                    &format!("Software\\Classes\\{p}\\DefaultIcon"),
                    None,
                    &doc_icon(&exe),
                )?;
                decorated.push((ext, p));
            }
        }
    }
    notify_shell();

    println!("registered {dll}");
    println!("  class    HKCU\\{key}");
    for ext in THUMBNAILED {
        match taken.iter().find(|(e, _)| *e == ext) {
            Some((_, other)) => println!("  {ext:<8} left to {other}, which already draws it"),
            None => println!("  {ext:<8} HKCU\\{}", shellex_key(ext)),
        }
    }
    println!("  progid   HKCU\\{progid}");
    for (ext, p) in &decorated {
        println!("  icon     HKCU\\Software\\Classes\\{p}\\DefaultIcon  (opens {ext})");
    }
    if decorated.is_empty() {
        println!(
            "\nNo file type resolves to this executable yet, so STEP files keep\n\
             whatever icon they have. Open one with stpview once -- Open with ->\n\
             Always -- and run --register again to give the type its document icon."
        );
    }
    print_caveats();
    Ok(())
}

pub fn unregister() -> Result<(), String> {
    // Only keys naming this provider; `register` leaves others in place.
    let clsid = stpview_shell::clsid_string();
    for ext in THUMBNAILED {
        let ours = read_string(HKEY_CURRENT_USER, &shellex_key(ext), None)
            .is_some_and(|v| v.eq_ignore_ascii_case(&clsid));
        if ours {
            delete_tree(&shellex_key(ext))?;
        }
    }
    delete_tree(&clsid_key())?;

    let exe = exe_path()?;
    delete_tree(&format!("Software\\Classes\\{PROGID}"))?;
    for ext in EXTENSIONS {
        delete_value(&format!("Software\\Classes\\{ext}\\OpenWithProgids"), PROGID)?;
        // Only an icon naming this executable, i.e. one `register` wrote;
        // other applications' ProgIDs keep theirs.
        for p in icon_progids(ext) {
            let ours = read_class(&format!("{p}\\DefaultIcon"), None)
                .map(|v| v.to_ascii_lowercase().contains(&exe.to_ascii_lowercase()))
                .unwrap_or(false);
            if ours {
                delete_tree(&format!("Software\\Classes\\{p}\\DefaultIcon"))?;
            }
        }
    }
    notify_shell();

    println!("unregistered; {} is no longer referenced", DLL);
    println!(
        "  a dllhost.exe that already loaded it keeps it open until that process\n  \
         retires, which is also what holds the file against a rebuild"
    );
    Ok(())
}

/// The two things that make a correct registration look broken.
fn print_caveats() {
    println!(
        "\nExplorer caches thumbnails by path, size and write time, so files it has\n\
         already drawn keep their old icon. Open a folder of models it has not\n\
         seen, or clear the cache, to see this working."
    );
    println!(
        "The handler is hosted in dllhost.exe, which holds the DLL open: kill that\n\
         process before rebuilding, or the link step fails with the file in use."
    );
}

