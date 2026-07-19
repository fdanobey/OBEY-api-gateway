fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../Assets/icon.ico");

    #[cfg(target_os = "windows")]
    compile_windows_resources();
}

#[cfg(target_os = "windows")]
fn compile_windows_resources() {
    let manifest_dir = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let icon_path = manifest_dir.join("../../Assets/icon.ico");

    let mut resource = winres::WindowsResource::new();
    resource.set_icon(icon_path.to_string_lossy().as_ref());
    resource.set("ProductName", env!("CARGO_PKG_NAME"));
    resource.set(
        "FileDescription",
        "OBEY API Gateway with optional system tray support",
    );
    resource.set("OriginalFilename", "ai-gateway.exe");
    resource.set("ProductVersion", env!("CARGO_PKG_VERSION"));
    resource.set("FileVersion", env!("CARGO_PKG_VERSION"));

    resource
        .compile()
        .expect("failed to compile Windows resources");
}
