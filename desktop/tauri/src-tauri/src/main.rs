use std::{
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
};
use tauri::Manager;

struct BackendProcess(Mutex<Option<Child>>);

fn main() {
    tauri::Builder::default()
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
            let log_file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_dir.join("backend.log"))?;
            let error_file = log_file.try_clone()?;

            let child = Command::new(backend_path)
                .args(["--port", "18731"])
                .stdin(Stdio::null())
                .stdout(Stdio::from(log_file))
                .stderr(Stdio::from(error_file))
                .spawn()?;
            app.manage(BackendProcess(Mutex::new(Some(child))));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error al iniciar MarketVision AI")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                let process = app.state::<BackendProcess>();
                let mut child = match process.0.lock() {
                    Ok(mut process) => process.take(),
                    Err(error) => {
                        eprintln!("Could not access the MarketVision backend process: {error}");
                        None
                    }
                };
                if let Some(mut child) = child.take() {
                    match child.try_wait() {
                        Ok(Some(_)) => {}
                        Ok(None) => {
                            if let Err(error) = child.kill() {
                                eprintln!("Could not stop the MarketVision backend: {error}");
                            }
                            if let Err(error) = child.wait() {
                                eprintln!("Could not wait for the MarketVision backend: {error}");
                            }
                        }
                        Err(error) => {
                            eprintln!("Could not inspect the MarketVision backend: {error}");
                        }
                    }
                }
            }
        });
}
