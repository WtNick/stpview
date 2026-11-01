//! 3Dconnexion SpaceMouse, read over raw HID.
//!
//! No 3Dconnexion SDK: the motion reports live on the HID interface declaring
//! Generic Desktop / Multi-Axis Controller. Coexists with the 3DxWare driver.
//!
//! Reading blocks, so a thread publishes the latest six axes into a shared
//! cell, which the viewer reads whenever it draws.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hidapi::{DeviceInfo, HidApi};

use stpview::camera::Motion;

/// 3Dconnexion's own USB vendor id. Older pucks, and the universal receiver on
/// some firmware, still enumerate under Logitech's.
const VID_3DCONNEXION: u16 = 0x256f;
const VID_LOGITECH: u16 = 0x046d;

/// HID usage identifying the interface that carries the motion data.
const USAGE_PAGE_GENERIC_DESKTOP: u16 = 0x01;
const USAGE_MULTI_AXIS_CONTROLLER: u16 = 0x08;

/// Report ids used by the 3Dconnexion HID protocol.
const REPORT_TRANSLATION: u8 = 0x01;
const REPORT_ROTATION: u8 = 0x02;
const REPORT_BUTTONS: u8 = 0x03;

/// Deflection at which an axis reads full scale. Current devices clamp at 350.
const FULL_SCALE: f64 = 350.0;

/// Deflection below which an axis is treated as centred, as a fraction of full
/// scale, so an imperfectly centred puck does not drift the view. Motion is
/// rescaled from the edge of the dead zone, not clamped, so there is no step.
const DEAD_ZONE: f64 = 0.04;

/// Fraction of the hardest-pushed axis below which an axis is treated as
/// spill from that one and dropped.
///
/// Tilting the cap also pushes it, so a tip would otherwise zoom as well.
/// Like 3DxWare's dominant axis, but proportional, so a deliberate second
/// gesture (pan while zooming) survives.
const CROSSTALK: f64 = 0.35;

/// How long a deflection may go unreported before it is treated as released.
/// The device streams while off centre and sends one zeroed report on release;
/// this covers that report going missing (a wireless dropout), which would
/// otherwise spin the camera forever.
const RELEASE_TIMEOUT: Duration = Duration::from_millis(1000);

/// Zeroes every axis well below the one being pushed hardest; see
/// [`CROSSTALK`]. A free function because [`Motion`] belongs to the library,
/// while this shaping is specific to the device.
fn drop_crosstalk(m: &mut Motion) {
    let peak = m
        .translate
        .iter()
        .chain(&m.rotate)
        .fold(0.0f64, |x, v| x.max(v.abs()));
    if peak == 0.0 {
        return;
    }
    let floor = peak * CROSSTALK;
    for x in m.translate.iter_mut().chain(m.rotate.iter_mut()) {
        if x.abs() < floor {
            *x = 0.0;
        }
    }
}

/// The six axes plus buttons, exactly as the device last reported them.
#[derive(Default, Clone, Copy, PartialEq, Debug)]
struct Axes {
    /// Right, forward, up.
    translation: [i16; 3],
    /// About X, Y, Z -- pitch, roll, yaw.
    rotation: [i16; 3],
    buttons: u32,
}

impl Axes {
    /// Folds one input report into the state. Returns false for report ids
    /// not interpreted (battery level, link status and the like).
    fn apply(&mut self, report: &[u8]) -> bool {
        let Some((&id, data)) = report.split_first() else {
            return false;
        };
        match id {
            REPORT_TRANSLATION if data.len() >= 6 => {
                self.translation = read_axes(data);
                // Wireless and newer wired devices pack all six axes into one
                // report rather than sending 0x01 and 0x02 in turn.
                if data.len() >= 12 {
                    self.rotation = read_axes(&data[6..]);
                }
                true
            }
            REPORT_ROTATION if data.len() >= 6 => {
                self.rotation = read_axes(data);
                true
            }
            REPORT_BUTTONS => {
                let mut mask = 0u32;
                for (i, byte) in data.iter().take(4).enumerate() {
                    mask |= (*byte as u32) << (8 * i);
                }
                self.buttons = mask;
                true
            }
            _ => false,
        }
    }

    fn to_motion(self) -> Motion {
        let f = |raw: i16| response(raw as f64 / FULL_SCALE);
        let mut m = Motion {
            translate: self.translation.map(f),
            rotate: self.rotation.map(f),
        };
        drop_crosstalk(&mut m);
        m
    }
}

/// Reads three little-endian signed 16-bit axis values.
fn read_axes(data: &[u8]) -> [i16; 3] {
    [
        i16::from_le_bytes([data[0], data[1]]),
        i16::from_le_bytes([data[2], data[3]]),
        i16::from_le_bytes([data[4], data[5]]),
    ]
}

/// Dead zone, then a curve that keeps fine control near the centre.
///
/// Cubic-weighted rather than a pure cube, so small deflections still move
/// the view noticeably.
fn response(x: f64) -> f64 {
    let mag = x.abs();
    if mag <= DEAD_ZONE {
        return 0.0;
    }
    // Rescaled from the edge of the dead zone: full deflection is still 1.0.
    let t = ((mag - DEAD_ZONE) / (1.0 - DEAD_ZONE)).min(1.0);
    x.signum() * t * (0.3 + 0.7 * t * t)
}

/// State shared between the reader thread and the viewer.
struct Shared {
    axes: Mutex<(Axes, Instant)>,
    /// Buttons pressed since the viewer last looked, latched so a press between
    /// two frames is never missed.
    pressed: Mutex<u32>,
    stop: AtomicBool,
}

/// An open SpaceMouse, being read on a thread of its own.
pub struct SpaceMouse {
    name: String,
    shared: Arc<Shared>,
}

impl Drop for SpaceMouse {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
    }
}

impl SpaceMouse {
    /// Opens the first SpaceMouse found, or returns why it could not.
    ///
    /// `wake` is called from the reader thread whenever a report arrives, so
    /// the event loop, asleep in `Wait`, notices the device.
    pub fn open(wake: impl Fn() + Send + 'static) -> Result<SpaceMouse, String> {
        let api = HidApi::new().map_err(|e| format!("could not initialise HID: {e}"))?;
        let info = find(&api).ok_or("no SpaceMouse found")?;
        let name = describe(&info);
        let device = api.open_path(info.path()).map_err(|e| {
            format!(
                "could not open {name} ({e}); \
                 3DxWare may be holding it open -- quit it from its tray icon"
            )
        })?;

        let shared = Arc::new(Shared {
            axes: Mutex::new((Axes::default(), Instant::now())),
            pressed: Mutex::new(0),
            stop: AtomicBool::new(false),
        });
        let worker = Arc::clone(&shared);
        std::thread::spawn(move || {
            let mut state = Axes::default();
            let mut buf = [0u8; 64];
            while !worker.stop.load(Ordering::Relaxed) {
                // A timeout rather than a blocking read, so dropping the
                // SpaceMouse ends the thread instead of leaking it.
                match device.read_timeout(&mut buf, 100) {
                    Ok(0) => continue,
                    Ok(n) => {
                        let before = state.buttons;
                        if !state.apply(&buf[..n]) {
                            continue;
                        }
                        // Edge triggered, so a held button fires once.
                        let fresh = state.buttons & !before;
                        if fresh != 0 {
                            *worker.pressed.lock().unwrap() |= fresh;
                        }
                        *worker.axes.lock().unwrap() = (state, Instant::now());
                        wake();
                    }
                    Err(_) => break, // unplugged; the viewer carries on without it
                }
            }
        });

        Ok(SpaceMouse { name, shared })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The current six axes, normalised and shaped.
    pub fn motion(&self) -> Motion {
        let (axes, at) = *self.shared.axes.lock().unwrap();
        if at.elapsed() > RELEASE_TIMEOUT {
            return Motion::default();
        }
        axes.to_motion()
    }

    /// Buttons pressed since this was last called, as a bitmask.
    pub fn take_pressed(&self) -> u32 {
        std::mem::take(&mut *self.shared.pressed.lock().unwrap())
    }

    /// The six axes as the device reported them, before the dead zone and the
    /// response curve, plus the buttons currently held.
    ///
    /// For `--puck`, which shows the raw numbers so axis signs can be checked.
    pub fn raw(&self) -> ([i16; 3], [i16; 3], u32) {
        let (a, _) = *self.shared.axes.lock().unwrap();
        (a.translation, a.rotation, a.buttons)
    }
}

/// Prints the raw axes and what the camera would make of them, until killed.
/// Tells a puck that is not being read from one mapped the wrong way round.
pub fn watch() -> Result<(), String> {
    let puck = SpaceMouse::open(|| {})?;
    println!("puck   {}", puck.name());
    println!(
        "       {:>7} {:>7} {:>7}   {:>7} {:>7} {:>7}",
        "right", "fwd", "up", "pitch", "roll", "twist"
    );
    println!("       push the puck; Ctrl+C to stop\n");

    let mut last = ([0i16; 3], [0i16; 3], 0u32);
    loop {
        let now = puck.raw();
        if now != last {
            let m = puck.motion();
            let (t, r, buttons) = now;
            println!(
                "raw    {:>7} {:>7} {:>7}   {:>7} {:>7} {:>7}{}",
                t[0],
                t[1],
                t[2],
                r[0],
                r[1],
                r[2],
                if buttons == 0 {
                    String::new()
                } else {
                    format!("   buttons {buttons:#x}")
                }
            );
            // Mark an axis the crosstalk filter dropped, so it is not
            // mistaken for one the device is not sending.
            let six: Vec<String> = t
                .iter()
                .chain(&r)
                .zip(m.translate.iter().chain(&m.rotate))
                .map(|(&raw, &out)| {
                    let curve = response(raw as f64 / FULL_SCALE);
                    if out == 0.0 && curve != 0.0 {
                        format!("{:>7}", "spill")
                    } else {
                        format!("{out:>7.3}")
                    }
                })
                .collect();
            println!(
                "camera {} {} {}   {} {} {}",
                six[0], six[1], six[2], six[3], six[4], six[5]
            );
            last = now;
        }
        // Fast enough to follow a hand, slow enough not to spin a core.
        std::thread::sleep(Duration::from_millis(30));
    }
}

fn is_motion_interface(info: &DeviceInfo) -> bool {
    // hidraw exposes one node per USB interface and does not surface usages,
    // so on Linux the usage fields read zero and the motion reports are on the
    // first interface instead.
    if info.usage_page() == 0 && info.usage() == 0 {
        return info.interface_number() <= 0;
    }
    info.usage_page() == USAGE_PAGE_GENERIC_DESKTOP && info.usage() == USAGE_MULTI_AXIS_CONTROLLER
}

fn find(api: &HidApi) -> Option<DeviceInfo> {
    api.device_list()
        .find(|i| {
            matches!(i.vendor_id(), VID_3DCONNEXION | VID_LOGITECH) && is_motion_interface(i)
        })
        .cloned()
}

fn describe(info: &DeviceInfo) -> String {
    format!(
        "{} ({:04x}:{:04x})",
        info.product_string().unwrap_or("unknown device"),
        info.vendor_id(),
        info.product_id()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_reports_update_each_half_of_the_state() {
        let mut a = Axes::default();
        assert!(a.apply(&[REPORT_TRANSLATION, 0xfe, 0xff, 0x03, 0x00, 0xfc, 0xff]));
        assert_eq!(a.translation, [-2, 3, -4]);
        assert_eq!(a.rotation, [0, 0, 0]);

        assert!(a.apply(&[REPORT_ROTATION, 0x05, 0x00, 0xfa, 0xff, 0x07, 0x00]));
        assert_eq!(a.translation, [-2, 3, -4], "rotation must not clobber");
        assert_eq!(a.rotation, [5, -6, 7]);
    }

    #[test]
    fn a_combined_report_carries_all_six_axes() {
        let mut a = Axes::default();
        assert!(a.apply(&[
            REPORT_TRANSLATION, //
            0x01, 0x00, 0x02, 0x00, 0x03, 0x00, // translation 1, 2, 3
            0xff, 0xff, 0xfe, 0xff, 0xfd, 0xff, // rotation -1, -2, -3
        ]));
        assert_eq!(a.translation, [1, 2, 3]);
        assert_eq!(a.rotation, [-1, -2, -3]);
    }

    #[test]
    fn reports_we_do_not_understand_leave_the_state_alone() {
        let mut a = Axes::default();
        assert!(a.apply(&[REPORT_BUTTONS, 0x03, 0x01]));
        assert_eq!(a.buttons, 0x0103);
        assert!(!a.apply(&[0x17, 0x64]), "battery level is not motion data");
        assert!(!a.apply(&[]));
        assert!(!a.apply(&[REPORT_TRANSLATION, 0x01]), "short report");
        assert_eq!(a.buttons, 0x0103, "state survives ignored reports");
        assert_eq!(a.translation, [0, 0, 0]);
    }

    /// The response curve is odd, monotonic, inclusive of the dead zone and
    /// saturates at exactly 1.0.
    #[test]
    fn the_response_curve_is_odd_monotonic_and_reaches_full_speed() {
        assert_eq!(response(0.0), 0.0);
        assert_eq!(response(DEAD_ZONE), 0.0, "the dead zone is inclusive");
        assert_eq!(response(-DEAD_ZONE), 0.0);

        // Full deflection means full speed: a curve that tops out below 1.0
        // would quietly rescale every rate constant in the camera.
        assert!((response(1.0) - 1.0).abs() < 1e-12);
        assert!((response(-1.0) + 1.0).abs() < 1e-12);
        // And past full scale it saturates rather than accelerating.
        assert!((response(2.5) - 1.0).abs() < 1e-12);

        let mut last = 0.0;
        for i in 0..=1000 {
            let x = i as f64 / 1000.0;
            let y = response(x);
            assert!(y >= last - 1e-15, "not monotonic at {x}");
            assert!((0.0..=1.0).contains(&y), "{y} out of range at {x}");
            assert!(
                (response(-x) + y).abs() < 1e-15,
                "not odd symmetric at {x}"
            );
            last = y;
        }
    }

    /// No step where the dead zone ends.
    #[test]
    fn the_response_curve_is_continuous_where_the_dead_zone_ends() {
        let eps = 1e-9;
        assert!(
            response(DEAD_ZONE + eps).abs() < 1e-6,
            "a jump at the dead zone edge: {}",
            response(DEAD_ZONE + eps)
        );
    }

    /// A tip leaks into the forward push; the spill is dropped, while two
    /// comparable gestures both stand.
    #[test]
    fn a_dominant_axis_drowns_out_the_spill_from_it() {
        let full = FULL_SCALE as i16;
        let tip = Axes {
            // Tipping hard about +X, with a fifth of that leaking into the
            // forward push that drives the zoom.
            translation: [0, full / 5, 0],
            rotation: [full, 0, 0],
            buttons: 0,
        };
        let m = tip.to_motion();
        assert!(m.rotate[0].abs() > 0.9, "the gesture itself must survive");
        assert_eq!(m.translate[1], 0.0, "the spill must not");

        // Two axes pushed comparably are two gestures, and both stand.
        let both = Axes {
            translation: [0, full, 0],
            rotation: [(FULL_SCALE * 0.8) as i16, 0, 0],
            buttons: 0,
        };
        let m = both.to_motion();
        assert!(m.translate[1].abs() > 0.5 && m.rotate[0].abs() > 0.5, "{m:?}");

        // A single axis is never its own crosstalk.
        for i in 0..3 {
            let mut t = [0i16; 3];
            t[i] = full / 3;
            let m = Axes { translation: t, rotation: [0; 3], buttons: 0 }.to_motion();
            assert!(m.translate[i] != 0.0, "axis {i} suppressed itself");
        }
    }

    #[test]
    fn a_centred_puck_is_still() {
        assert!(Axes::default().to_motion().is_still());
        // Inside the dead zone on every axis is also still.
        let noise = (DEAD_ZONE * FULL_SCALE) as i16;
        let jitter = Axes {
            translation: [noise, -noise, noise],
            rotation: [-noise, noise, -noise],
            buttons: 0,
        };
        assert!(jitter.to_motion().is_still(), "{:?}", jitter.to_motion());
        // And one count past it is not.
        let moved = Axes {
            translation: [0, 0, 0],
            rotation: [0, 0, (FULL_SCALE * 0.5) as i16],
            buttons: 0,
        };
        assert!(!moved.to_motion().is_still());
    }
}
