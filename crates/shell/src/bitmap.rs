//! The rendered pixels as the bitmap the shell takes ownership of.

use windows::Win32::Foundation::E_FAIL;
use windows::Win32::Graphics::Gdi::{
    BITMAPINFO, BITMAPINFOHEADER, BI_RGB, CreateDIBSection, DIB_RGB_COLORS, HBITMAP,
};

/// Rewrites tightly packed RGBA8 as the BGRA a 32-bit DIB is read as.
///
/// Alpha is set opaque rather than copied: the thumbnail is handed over as
/// `WTSAT_RGB`, and opaque bytes keep a consumer that ignores that flag from
/// blending the picture away.
pub fn bgra_from_rgba(rgba: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; rgba.len() / 4 * 4];
    for (dst, src) in out.chunks_exact_mut(4).zip(rgba.chunks_exact(4)) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst[3] = 255;
    }
    out
}

/// A 32-bit top-down DIB section holding `bgra`, for the caller to hand to the
/// shell. Ownership goes with it: `IThumbnailProvider::GetThumbnail` returns a
/// bitmap the shell then deletes.
///
/// # Safety
///
/// `bgra` must hold exactly `width * height * 4` bytes.
pub unsafe fn hbitmap(width: i32, height: i32, bgra: &[u8]) -> windows::core::Result<HBITMAP> {
    let info = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width,
            // Negative: top-down rows, as `capture` returns them. A DIB
            // defaults to bottom-up, which would silently flip the picture.
            biHeight: -height,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
    let bmp = unsafe { CreateDIBSection(None, &info, DIB_RGB_COLORS, &mut bits, None, 0)? };
    if bits.is_null() {
        return Err(E_FAIL.into());
    }
    // A `width * 4` stride is already DIB-aligned, so one copy suffices.
    unsafe { std::ptr::copy_nonoverlapping(bgra.as_ptr(), bits as *mut u8, bgra.len()) };
    Ok(bmp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_becomes_opaque_bgra() {
        // The second pixel's alpha must not survive.
        let rgba = [10u8, 20, 30, 255, 40, 50, 60, 7];
        assert_eq!(bgra_from_rgba(&rgba), vec![30, 20, 10, 255, 60, 50, 40, 255]);
    }

    #[test]
    fn a_partial_pixel_is_dropped_rather_than_read_past() {
        assert_eq!(bgra_from_rgba(&[1, 2, 3, 4, 9, 9]).len(), 4);
    }
}
