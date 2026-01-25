//! GPU Shell - Interactive command line with GPU-accelerated pipelines
//!
//! A PowerShell-style shell where pipes pass GPU buffers instead of text.
//!
//! Usage:
//!   cargo run --release --example gpu_shell
//!
//! Example commands:
//!   files ~/code | where ext = "rs" | count
//!   files ~ | where size > 100MB | sort size desc | head 10
//!   search "TODO" ~/code | head 20
//!   files . | group ext | sort count desc

use std::io::{self, Write, BufRead};
use std::time::Instant;

use rust_experiment::gpu_os::shell::{GpuShell, Value, TableRenderer};

fn main() {
    println!("GPU Shell v0.1");
    println!("==============");

    // Initialize shell
    let mut shell = match GpuShell::new() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to initialize GPU Shell: {}", e);
            std::process::exit(1);
        }
    };

    println!("GPU initialized. Type 'help' for commands, 'exit' to quit.\n");

    let renderer = TableRenderer::default();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Prompt
        print!("gpu> ");
        stdout.flush().unwrap();

        // Read input
        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();

        // Handle special commands
        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" || input == "q" {
            println!("Goodbye!");
            break;
        }

        if input == "clear" || input == "cls" {
            print!("\x1B[2J\x1B[1;1H");
            stdout.flush().unwrap();
            continue;
        }

        if input == "stats" {
            let (hits, misses, cached, index_hits) = shell.cache_stats();
            println!("Cache: {} hits, {} misses, {} cached paths", hits, misses, cached);
            println!("GPU Index: {} index loads", index_hits);
            if let Some((entries, mem)) = shell.gpu_index_info() {
                println!("GPU Index: {} entries, {:.1} MB", entries, mem as f64 / 1024.0 / 1024.0);
            }
            continue;
        }

        if input == "index rebuild" {
            match shell.rebuild_index() {
                Ok(()) => println!("Index rebuilt successfully"),
                Err(e) => println!("Error: {}", e),
            }
            continue;
        }

        if input == "index" || input == "index info" {
            if let Some((entries, mem)) = shell.gpu_index_info() {
                println!("GPU Filesystem Index");
                println!("  Entries: {}", entries);
                println!("  Memory:  {:.1} MB", mem as f64 / 1024.0 / 1024.0);
                println!("  Status:  Active");
            } else {
                println!("GPU Index: Not loaded (using filesystem scan)");
            }
            continue;
        }

        if input == "cache clear" {
            shell.clear_cache();
            println!("Cache cleared");
            continue;
        }

        // Execute command
        let start = Instant::now();

        match shell.execute(input) {
            Ok(result) => {
                let output = renderer.render(&result, None);
                println!("{}", output);

                // Show timing
                let elapsed = start.elapsed();
                if elapsed.as_millis() < 1 {
                    println!("({:.1}µs)", elapsed.as_micros() as f64);
                } else {
                    println!("({:.1}ms)", elapsed.as_secs_f64() * 1000.0);
                }
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }

        println!();
    }
}
