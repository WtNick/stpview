//! Embeds the application icon, the STEP document icon and a version block.
//!
//! Without an icon resource Windows gives the exe the generic application
//! icon, and Explorer falls back to that for STEP files with no `DefaultIcon`
//! of their own, badging every thumbnail as a program.
//!
//! The icons under `icons/` are drawn by `examples/icon.rs`; this only
//! packages them.

fn main() {
    println!("cargo:rerun-if-changed=icons/stpview.ico");
    println!("cargo:rerun-if-changed=icons/step.ico");
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        // Resource id 1 is the application icon, because Windows shows the
        // lowest-numbered icon for the executable itself. Id 2 is the document
        // icon, which nothing finds on its own -- `--register` names it
        // explicitly, as `<exe>,-2`.
        res.set_icon_with_id("icons/stpview.ico", "1");
        res.set_icon_with_id("icons/step.ico", "2");
        res.set("FileDescription", "stpview - STEP model viewer");
        res.set("ProductName", "stpview");
        res.set("OriginalFilename", "stpview.exe");
        if let Err(e) = res.compile() {
            // Not fatal: without the Windows SDK's resource compiler the viewer
            // still works, with the generic icon.
            println!("cargo:warning=no icon embedded: {e}");
        }
    }
}
