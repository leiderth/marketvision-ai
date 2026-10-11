#[cfg(windows)]
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::{
    fs::File,
    io::Write,
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::JoinHandle,
};
use tauri::{Emitter, Manager, State};

struct BackendProcess {
    child: Mutex<Option<Child>>,
    lifecycle_log: Arc<Mutex<File>>,
    #[cfg(windows)]
    _job: BackendJob,
}

#[cfg(windows)]
struct BackendJob {
    handle: OwnedHandle,
}

#[cfg(windows)]
impl BackendJob {
    fn new() -> std::io::Result<Self> {
        use std::{ffi::c_void, mem::size_of, ptr::null};
        use windows_sys::Win32::{
            Foundation::GetLastError,
            System::JobObjects::{
                CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            },
        };

        let handle = unsafe { CreateJobObjectW(null(), null()) };
        if handle.is_null() {
            return Err(std::io::Error::from_raw_os_error(
                unsafe { GetLastError() } as i32
            ));
        }

        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let configured = unsafe {
            SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const c_void,
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if configured == 0 {
            let error = unsafe { GetLastError() } as i32;
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(handle);
            }
            return Err(std::io::Error::from_raw_os_error(error));
        }
        Ok(Self {
            handle: unsafe { OwnedHandle::from_raw_handle(handle as _) },
        })
    }

    fn assign(&self, child: &Child) -> std::io::Result<()> {
        use windows_sys::Win32::{
            Foundation::{GetLastError, HANDLE},
            System::JobObjects::AssignProcessToJobObject,
        };

        let job = self.handle.as_raw_handle() as HANDLE;
        let process = child.as_raw_handle() as HANDLE;
        if unsafe { AssignProcessToJobObject(job, process) } == 0 {
            return Err(std::io::Error::from_raw_os_error(
                unsafe { GetLastError() } as i32
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct InteractionRect {
    left: f64,
    top: f64,
    width: f64,
    height: f64,
}

struct OverlayController {
    click_through: Arc<AtomicBool>,
    interaction_rect: Arc<Mutex<Option<InteractionRect>>>,
    lifecycle_log: Arc<Mutex<File>>,
    #[cfg(windows)]
    hotkey_thread_id: Arc<std::sync::atomic::AtomicU32>,
    #[cfg(windows)]
    hotkey_thread: Mutex<Option<JoinHandle<()>>>,
}

impl OverlayController {
    fn new(lifecycle_log: Arc<Mutex<File>>) -> Self {
        Self {
            click_through: Arc::new(AtomicBool::new(false)),
            interaction_rect: Arc::new(Mutex::new(None)),
            lifecycle_log,
            #[cfg(windows)]
            hotkey_thread_id: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            #[cfg(windows)]
            hotkey_thread: Mutex::new(None),
        }
    }

    #[cfg(windows)]
    fn stop_hotkeys(&self) {
        use windows_sys::Win32::UI::WindowsAndMessaging::{PostThreadMessageW, WM_QUIT};

        let thread_id = self.hotkey_thread_id.load(Ordering::SeqCst);
        if thread_id != 0 {
            let posted = unsafe { PostThreadMessageW(thread_id, WM_QUIT, 0, 0) };
            if posted == 0 {
                log_lifecycle_error(
                    &self.lifecycle_log,
                    &format!(
                        "Could not stop the global shortcut listener (Windows error {}).",
                        unsafe { windows_sys::Win32::Foundation::GetLastError() }
                    ),
                );
            }
        }
        match self.hotkey_thread.lock() {
            Ok(mut thread) => {
                if let Some(thread) = thread.take() {
                    if thread.join().is_err() {
                        log_lifecycle_error(
                            &self.lifecycle_log,
                            "The global shortcut listener did not exit cleanly.",
                        );
                    }
                }
            }
            Err(error) => log_lifecycle_error(
                &self.lifecycle_log,
                &format!("Could not access the global shortcut listener: {error}"),
            ),
        }
    }
}

#[tauri::command]
fn set_click_through(
    window: tauri::WebviewWindow,
    controller: State<'_, OverlayController>,
    enabled: bool,
) -> Result<(), String> {
    apply_click_through(&window, &controller, enabled)
}

#[tauri::command]
fn set_overlay_interaction_region(
    window: tauri::WebviewWindow,
    controller: State<'_, OverlayController>,
    left: f64,
    top: f64,
    width: f64,
    height: f64,
) -> Result<(), String> {
    if ![left, top, width, height]
        .iter()
        .all(|value| value.is_finite())
        || left < 0.0
        || top < 0.0
        || width <= 0.0
        || height <= 0.0
    {
        return Err("The overlay interaction panel has invalid dimensions.".to_string());
    }

    let rect = InteractionRect {
        left,
        top,
        width,
        height,
    };
    *controller
        .interaction_rect
        .lock()
        .map_err(|error| format!("Could not store the overlay interaction region: {error}"))? =
        Some(rect);

    #[cfg(windows)]
    if controller.click_through.load(Ordering::SeqCst) {
        apply_native_interaction_region(&window, Some(rect))?;
    }
    Ok(())
}

fn log_lifecycle_error(log_file: &Arc<Mutex<File>>, message: &str) {
    if let Ok(mut log) = log_file.lock() {
        let _ = writeln!(log, "{message}");
    }
}

fn apply_click_through(
    window: &tauri::WebviewWindow,
    controller: &OverlayController,
    enabled: bool,
) -> Result<(), String> {
    #[cfg(windows)]
    {
        let rect = if enabled {
            Some(
                controller
                    .interaction_rect
                    .lock()
                    .map_err(|error| {
                        format!("Could not read the overlay interaction region: {error}")
                    })?
                    .ok_or_else(|| {
                        "The overlay panel has not reported its interaction region yet.".to_string()
                    })?,
            )
        } else {
            None
        };
        apply_native_interaction_region(window, rect)?;
    }
    #[cfg(not(windows))]
    window
        .set_ignore_cursor_events(enabled)
        .map_err(|error| format!("Could not change click-through mode: {error}"))?;

    controller.click_through.store(enabled, Ordering::SeqCst);
    Ok(())
}

#[cfg(windows)]
fn apply_native_interaction_region(
    window: &tauri::WebviewWindow,
    rect: Option<InteractionRect>,
) -> Result<(), String> {
    use std::ptr::null_mut;
    use windows_sys::Win32::{
        Foundation::{GetLastError, HWND},
        Graphics::Gdi::{CreateRectRgn, DeleteObject, SetWindowRgn},
    };

    let hwnd = window
        .hwnd()
        .map_err(|error| format!("Could not get the overlay window handle: {error}"))?
        .0 as HWND;

    let Some(rect) = rect else {
        let result = unsafe { SetWindowRgn(hwnd, null_mut(), 1) };
        return if result == 0 {
            Err(format!(
                "Could not restore the overlay input region (Windows error {}).",
                unsafe { GetLastError() }
            ))
        } else {
            Ok(())
        };
    };

    let scale = window
        .scale_factor()
        .map_err(|error| format!("Could not read the overlay DPI scale: {error}"))?;
    let left = (rect.left * scale).round() as i32;
    let top = (rect.top * scale).round() as i32;
    let right = ((rect.left + rect.width) * scale).round() as i32;
    let bottom = ((rect.top + rect.height) * scale).round() as i32;
    if right <= left || bottom <= top {
        return Err("The overlay interaction region is empty after DPI scaling.".to_string());
    }

    let region = unsafe { CreateRectRgn(left, top, right, bottom) };
    if region.is_null() {
        return Err(format!(
            "Could not create the overlay interaction region (Windows error {}).",
            unsafe { GetLastError() }
        ));
    }
    let result = unsafe { SetWindowRgn(hwnd, region, 1) };
    if result == 0 {
        let error = unsafe { GetLastError() };
        unsafe {
            DeleteObject(region);
        }
        return Err(format!(
            "Could not apply the overlay interaction region (Windows error {error})."
        ));
    }
    Ok(())
}

fn expand_to_virtual_desktop(window: &tauri::WebviewWindow) -> tauri::Result<()> {
    let monitors = window.available_monitors()?;
    let monitor_bounds = monitors.into_iter().map(|monitor| {
        (
            monitor.position().x,
            monitor.position().y,
            monitor.size().width as i32,
            monitor.size().height as i32,
        )
    });
    let (left, top, right, bottom) = virtual_desktop_bounds(monitor_bounds).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Windows did not report any monitors for the overlay.",
        )
    })?;

    let width = u32::try_from(right - left)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;
    let height = u32::try_from(bottom - top)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;

    window.set_position(tauri::PhysicalPosition::new(left, top))?;
    window.set_size(tauri::PhysicalSize::new(width, height))?;
    window.set_always_on_top(true)?;
    Ok(())
}

fn virtual_desktop_bounds(
    monitors: impl IntoIterator<Item = (i32, i32, i32, i32)>,
) -> Option<(i32, i32, i32, i32)> {
    let mut bounds: Option<(i32, i32, i32, i32)> = None;
    for (x, y, width, height) in monitors {
        let right = x.saturating_add(width);
        let bottom = y.saturating_add(height);
        bounds = Some(match bounds {
            Some((left, top, current_right, current_bottom)) => (
                left.min(x),
                top.min(y),
                current_right.max(right),
                current_bottom.max(bottom),
            ),
            None => (x, y, right, bottom),
        });
    }
    bounds
}

#[cfg(windows)]
fn start_global_hotkeys(
    app: tauri::AppHandle,
    controller: &OverlayController,
) -> std::io::Result<JoinHandle<()>> {
    use std::time::Duration;

    let click_through = Arc::clone(&controller.click_through);
    let interaction_rect = Arc::clone(&controller.interaction_rect);
    let thread_id = Arc::clone(&controller.hotkey_thread_id);
    let lifecycle_log = Arc::clone(&controller.lifecycle_log);
    let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);

    let thread = std::thread::Builder::new()
        .name("marketvision-global-hotkeys".to_string())
        .spawn(move || {
            use std::mem::zeroed;
            use std::ptr::null_mut;
            use windows_sys::Win32::{
                System::Threading::GetCurrentThreadId,
                UI::Input::KeyboardAndMouse::{RegisterHotKey, UnregisterHotKey, MOD_NOREPEAT},
                UI::WindowsAndMessaging::{
                    GetMessageW, PeekMessageW, MSG, PM_NOREMOVE, WM_HOTKEY, WM_QUIT,
                },
            };

            const HOTKEY_TOGGLE_OVERLAY: i32 = 1;
            const HOTKEY_TOGGLE_CLICK_THROUGH: i32 = 2;
            const HOTKEY_KILL_SWITCH: i32 = 3;
            const VK_F9: u32 = 0x78;
            const VK_F10: u32 = 0x79;
            const VK_F12: u32 = 0x7B;

            unsafe {
                let mut message: MSG = zeroed();
                PeekMessageW(&mut message, null_mut(), 0, 0, PM_NOREMOVE);
                thread_id.store(GetCurrentThreadId(), Ordering::SeqCst);
                let _ = ready_sender.send(());

                for (id, key, description) in [
                    (HOTKEY_TOGGLE_OVERLAY, VK_F9, "F9 overlay"),
                    (HOTKEY_TOGGLE_CLICK_THROUGH, VK_F10, "F10 click-through"),
                    (HOTKEY_KILL_SWITCH, VK_F12, "F12 kill switch"),
                ] {
                    if RegisterHotKey(null_mut(), id, MOD_NOREPEAT, key) == 0 {
                        log_lifecycle_error(
                            &lifecycle_log,
                            &format!(
                                "Could not register global {description} shortcut (Windows error {}).",
                                windows_sys::Win32::Foundation::GetLastError()
                            ),
                        );
                    }
                }

                loop {
                    let result = GetMessageW(&mut message, null_mut(), 0, 0);
                    if result < 0 {
                        log_lifecycle_error(
                            &lifecycle_log,
                            &format!(
                                "The global shortcut listener failed (Windows error {}).",
                                windows_sys::Win32::Foundation::GetLastError()
                            ),
                        );
                        break;
                    }
                    if result == 0 || message.message == WM_QUIT {
                        break;
                    }
                    if message.message != WM_HOTKEY {
                        continue;
                    }

                    let event_result = match message.wParam as i32 {
                        HOTKEY_TOGGLE_OVERLAY => {
                            if let Some(window) = app.get_webview_window("main") {
                                match window.is_visible() {
                                    Ok(true) => window.hide().map_err(|error| error.to_string()),
                                    Ok(false) => window.show().map_err(|error| error.to_string()),
                                    Err(error) => Err(error.to_string()),
                                }
                            } else {
                                Err("The main overlay window is unavailable.".to_string())
                            }
                        }
                        HOTKEY_TOGGLE_CLICK_THROUGH => {
                            if let Some(window) = app.get_webview_window("main") {
                                let enabled = !click_through.load(Ordering::SeqCst);
                                let region = if enabled {
                                    match interaction_rect.lock() {
                                        Ok(rect) => Ok(*rect),
                                        Err(error) => Err(format!(
                                            "Could not read the overlay interaction panel: {error}"
                                        )),
                                    }
                                } else {
                                    Ok(None)
                                };
                                let region_result = region.and_then(|rect| {
                                    if enabled && rect.is_none() {
                                        Err("The overlay has not reported its interaction panel.".to_string())
                                    } else {
                                        apply_native_interaction_region(&window, rect)
                                    }
                                });
                                region_result.and_then(|()| {
                                    click_through.store(enabled, Ordering::SeqCst);
                                    if !window.is_visible().map_err(|error| error.to_string())? {
                                        window.show().map_err(|error| error.to_string())?;
                                    }
                                    app.emit("marketvision-click-through", enabled)
                                        .map_err(|error| error.to_string())
                                })
                            } else {
                                Err("The main overlay window is unavailable.".to_string())
                            }
                        }
                        HOTKEY_KILL_SWITCH => {
                            if let Some(window) = app.get_webview_window("main") {
                                let region_result = apply_native_interaction_region(&window, None);
                                region_result.and_then(|()| {
                                    click_through.store(false, Ordering::SeqCst);
                                    if !window.is_visible().map_err(|error| error.to_string())? {
                                        window.show().map_err(|error| error.to_string())?;
                                    }
                                    app.emit("marketvision-click-through", false)
                                        .map_err(|error| error.to_string())?;
                                    app.emit("marketvision-kill-switch", ())
                                        .map_err(|error| error.to_string())
                                })
                            } else {
                                Err("The main overlay window is unavailable.".to_string())
                            }
                        }
                        _ => Ok(()),
                    };
                    if let Err(error) = event_result {
                        log_lifecycle_error(
                            &lifecycle_log,
                            &format!("Global shortcut action failed: {error}"),
                        );
                    }
                }

                for id in [
                    HOTKEY_TOGGLE_OVERLAY,
                    HOTKEY_TOGGLE_CLICK_THROUGH,
                    HOTKEY_KILL_SWITCH,
                ] {
                    UnregisterHotKey(null_mut(), id);
                }
                thread_id.store(0, Ordering::SeqCst);
            }
        })?;

    if ready_receiver.recv_timeout(Duration::from_secs(5)).is_err() {
        let id = controller.hotkey_thread_id.load(Ordering::SeqCst);
        if id != 0 {
            use windows_sys::Win32::UI::WindowsAndMessaging::{PostThreadMessageW, WM_QUIT};
            unsafe {
                PostThreadMessageW(id, WM_QUIT, 0, 0);
            }
        }
        let _ = thread.join();
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "The global shortcut listener did not initialize in time.",
        ));
    }
    Ok(thread)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            set_click_through,
            set_overlay_interaction_region
        ])
        .setup(|app| {
            let port_reservation = TcpListener::bind(("127.0.0.1", 18731))?;
            drop(port_reservation);

            let resource_dir = app.path().resource_dir()?;
            let backend_path = resource_dir
                .join("marketvision-backend")
                .join("marketvision-backend.exe");
            if !backend_path.is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "Packaged analysis backend not found: {}",
                        backend_path.display()
                    ),
                )
                .into());
            }

            let tesseract_dir = resource_dir.join("tesseract");
            let tesseract_path = tesseract_dir.join("tesseract.exe");
            let tessdata_path = tesseract_dir.join("tessdata");
            if !tesseract_path.is_file()
                || !tessdata_path.join("eng.traineddata").is_file()
                || !tessdata_path.join("configs").join("tsv").is_file()
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "Packaged offline OCR runtime is incomplete: {}",
                        tesseract_dir.display()
                    ),
                )
                .into());
            }

            let local_app_data = std::env::var_os("LOCALAPPDATA").ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "LOCALAPPDATA is required to store application logs.",
                )
            })?;
            let log_dir = PathBuf::from(local_app_data)
                .join("MarketVision AI")
                .join("logs");
            std::fs::create_dir_all(&log_dir)?;
            let crash_log = log_dir.join("crash.log");
            let previous_panic_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&crash_log)
                {
                    let _ = writeln!(file, "{info}");
                }
                previous_panic_hook(info);
            }));
            let log_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_dir.join("backend.log"))?;
            let error_file = log_file.try_clone()?;
            let lifecycle_log = Arc::new(Mutex::new(log_file.try_clone()?));

            #[cfg(windows)]
            let backend_job = BackendJob::new()?;
            let mut command = Command::new(backend_path);
            command
                .env("TESSERACT_CMD", &tesseract_path)
                .env("TESSDATA_DIR", &tessdata_path);
            #[cfg(windows)]
            command.creation_flags(0x08000000);
            let window = app.get_webview_window("main").ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Main overlay window not found.",
                )
            })?;
            expand_to_virtual_desktop(&window)?;
            let overlay = OverlayController::new(Arc::clone(&lifecycle_log));
            app.manage(overlay);
            #[cfg(windows)]
            {
                let controller = app.state::<OverlayController>();
                let hotkey_thread = start_global_hotkeys(app.handle().clone(), &controller)?;
                *controller.hotkey_thread.lock().map_err(|error| {
                    std::io::Error::new(std::io::ErrorKind::Other, error.to_string())
                })? = Some(hotkey_thread);
            }
            let child = match command
                .args(["--port", "18731"])
                .stdin(Stdio::null())
                .stdout(Stdio::from(log_file))
                .stderr(Stdio::from(error_file))
                .spawn()
            {
                Ok(child) => child,
                Err(error) => {
                    #[cfg(windows)]
                    app.state::<OverlayController>().stop_hotkeys();
                    return Err(error.into());
                }
            };
            #[cfg(windows)]
            if let Err(error) = backend_job.assign(&child) {
                let mut child = child;
                if let Err(kill_error) = child.kill() {
                    log_lifecycle_error(
                        &lifecycle_log,
                        &format!("Could not stop unassigned backend process: {kill_error}"),
                    );
                }
                if let Err(wait_error) = child.wait() {
                    log_lifecycle_error(
                        &lifecycle_log,
                        &format!("Could not wait for unassigned backend process: {wait_error}"),
                    );
                }
                return Err(error.into());
            }
            app.manage(BackendProcess {
                child: Mutex::new(Some(child)),
                lifecycle_log,
                #[cfg(windows)]
                _job: backend_job,
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error al iniciar MarketVision AI")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                #[cfg(windows)]
                app.state::<OverlayController>().stop_hotkeys();

                let process = app.state::<BackendProcess>();
                let mut child = match process.child.lock() {
                    Ok(mut child) => child.take(),
                    Err(error) => {
                        log_lifecycle_error(
                            &process.lifecycle_log,
                            &format!("Could not access the MarketVision backend process: {error}"),
                        );
                        None
                    }
                };
                if let Some(mut child) = child.take() {
                    match child.try_wait() {
                        Ok(Some(_)) => {}
                        Ok(None) => {
                            if let Err(error) = child.kill() {
                                log_lifecycle_error(
                                    &process.lifecycle_log,
                                    &format!("Could not stop the MarketVision backend: {error}"),
                                );
                            }
                            if let Err(error) = child.wait() {
                                log_lifecycle_error(
                                    &process.lifecycle_log,
                                    &format!(
                                        "Could not wait for the MarketVision backend: {error}"
                                    ),
                                );
                            }
                        }
                        Err(error) => {
                            log_lifecycle_error(
                                &process.lifecycle_log,
                                &format!("Could not inspect the MarketVision backend: {error}"),
                            );
                            if let Err(kill_error) = child.kill() {
                                log_lifecycle_error(
                                    &process.lifecycle_log,
                                    &format!(
                                        "Could not stop the MarketVision backend: {kill_error}"
                                    ),
                                );
                            }
                            if let Err(wait_error) = child.wait() {
                                log_lifecycle_error(
                                    &process.lifecycle_log,
                                    &format!(
                                        "Could not wait for the MarketVision backend: {wait_error}"
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::virtual_desktop_bounds;

    #[test]
    fn virtual_desktop_includes_negative_monitor_origins() {
        assert_eq!(
            virtual_desktop_bounds([(-1920, 0, 1920, 1080), (0, -200, 2560, 1440)]),
            Some((-1920, -200, 2560, 1240))
        );
    }
}
