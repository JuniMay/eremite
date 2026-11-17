//! Startup settings without model loading. An empty key or an occupied
//! listener stops startup before the engine opens a checkpoint.
use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

/// Returns the repository's folder: the folder above server/.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}

/// A home folder for one test, with a default config file in it. The folder
/// is removed when the test ends.
struct Home(PathBuf);

impl Home {
    /// Makes the home folder of the test `test`, with its default config file.
    fn new(test: &str) -> Home {
        let dir = std::env::temp_dir().join(format!("eremite-home-{test}-{}", std::process::id()));
        let folder = dir.join(".config/eremite");
        std::fs::create_dir_all(&folder).unwrap();
        let config = json!({"port": 8008, "host": "127.0.0.1", "api_key_file": "/dev/null",
                            "reserve": 10, "models": [{"model": "unused-model"}]});
        std::fs::write(folder.join("server.json"), config.to_string()).unwrap();
        Home(dir)
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs the server with `args` in the home folder of the test `test`. Writes
/// `config` to its standard input, if given. Returns its log, once it exited
/// before loading a model.
fn run(test: &str, args: &[String], config: Option<Value>) -> String {
    let home = Home::new(test);
    let mut child = Command::new(env!("CARGO_BIN_EXE_eremite-server"))
        .args(args)
        .current_dir(root().join("server"))
        .env("HOME", &home.0)
        .env_remove("XDG_CONFIG_HOME")
        .env("EREMITE_API_KEY", "startup-check-key-not-a-secret")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server starts");
    if let Some(config) = config {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(config.to_string().as_bytes())
            .unwrap();
    } else {
        drop(child.stdin.take());
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        !output.status.success(),
        "the server must stop before it loads a model"
    );
    String::from_utf8(output.stderr).unwrap()
}

#[test]
fn a_config_file_supplies_the_listener_and_key_file() {
    let config = json!({"port": 8008, "api_key_file": "/dev/null",
                        "models": [{"model": "unused-model"}]});
    let error = run(
        "config-file",
        &["--config".into(), "/dev/stdin".into()],
        Some(config),
    );
    assert!(error.contains("must have 16 to 256 characters"), "{error}");
}

#[test]
fn no_arguments_loads_the_default_home_config() {
    let error = run("no-arguments", &[], None);
    assert!(error.contains("must have 16 to 256 characters"), "{error}");
}

#[test]
fn command_line_key_file_overrides_the_default_config() {
    let error = run(
        "key-file-override",
        &["--api-key-file".into(), "missing-override-key".into()],
        None,
    );
    assert!(
        error.contains("cannot open missing-override-key"),
        "{error}"
    );
}

#[test]
fn explicit_config_and_cli_listener_override_stop_before_model_loading() {
    let guard = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = guard.local_addr().unwrap().port();
    let base = json!({"port": port, "host": "127.0.0.1",
                      "models": [{"model": "unused-model"}]});
    let error = run(
        "listener-override",
        &["--config".into(), "/dev/stdin".into()],
        Some(base.clone()),
    );
    assert!(
        error.contains(&format!("cannot listen on 127.0.0.1:{port}")),
        "{error}"
    );
    for before in [false, true] {
        let mut config = base.clone();
        config["port"] = json!(0);
        config["api_key_file"] = json!("missing-config-key");
        let cli = vec![
            "--serve".into(),
            port.to_string(),
            "--api-key-file".into(),
            "/dev/null".into(),
        ];
        let file = vec!["--config".into(), "/dev/stdin".into()];
        let args = if before {
            [cli, file].concat()
        } else {
            [file, cli].concat()
        };
        let error = run("listener-override", &args, Some(config));
        assert!(error.contains("must have 16 to 256 characters"), "{error}");
    }
    let mut config = base;
    config["port"] = json!(0);
    let error = run(
        "listener-override",
        &[
            "--config".into(),
            "/dev/stdin".into(),
            "--serve".into(),
            port.to_string(),
        ],
        Some(config),
    );
    assert!(
        error.contains(&format!("cannot listen on 127.0.0.1:{port}")),
        "{error}"
    );
}

#[test]
fn an_explicit_model_uses_cli_settings_without_the_default_file() {
    let guard = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = guard.local_addr().unwrap().port();
    let error = run(
        "explicit-model",
        &[
            "-m".into(),
            "unused-model".into(),
            "--serve".into(),
            port.to_string(),
        ],
        None,
    );
    assert!(
        error.contains(&format!("cannot listen on 127.0.0.1:{port}")),
        "{error}"
    );
}
