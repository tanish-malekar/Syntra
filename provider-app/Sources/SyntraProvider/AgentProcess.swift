import Foundation

/// Paths of the per-user install made by join.sh --install-only.
struct InstallLayout {
    let home: URL
    var python: URL { home.appendingPathComponent("venv/bin/python") }
    var agentDir: URL { home.appendingPathComponent("agent") }
    var bloombeeDir: URL { home.appendingPathComponent("BloomBee") }
    var isInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: python.path)
            && FileManager.default.fileExists(atPath: agentDir.appendingPathComponent("syntra_agent/cli.py").path)
            // an agent installed before the desktop app existed has no control API: treat it as outdated and refresh it
            && FileManager.default.fileExists(atPath: agentDir.appendingPathComponent("syntra_agent/control.py").path)
            && FileManager.default.fileExists(atPath: bloombeeDir.appendingPathComponent("src/bloombee").path)
    }
    var logFile: URL { home.appendingPathComponent("app-agent.log") }

    static var standard: InstallLayout {
        InstallLayout(home: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".syntra"))
    }
}

/// Output collected from a background reader thread.
private final class OutputBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""
    func append(_ s: String) { lock.lock(); text += s; lock.unlock() }
    var value: String { lock.lock(); defer { lock.unlock() }; return text }
}

/// Runs join.sh --install-only and reports each "==> step" line.
final class Installer {
    let server: URL
    let layout: InstallLayout
    init(server: URL, layout: InstallLayout) { self.server = server; self.layout = layout }

    /// Returns the full output on failure (shown in "Copy log").
    func run(onStep: @escaping (String) -> Void) async throws {
        let script = server.appendingPathComponent("join.sh").absoluteString
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        // Single-quote the URL for the shell; a URL never contains a single quote after percent-encoding.
        process.arguments = ["-c", "set -o pipefail; curl -fsSL '\(script)' | bash -s -- --install-only"]
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (env["PATH"] ?? "")
        env["SYNTRA_HOME"] = layout.home.path
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        let output = OutputBuffer()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = String(decoding: handle.availableData, as: UTF8.self)
            guard !chunk.isEmpty else { return }
            output.append(chunk)
            for line in chunk.split(separator: "\n") where line.hasPrefix("==> ") {
                onStep(String(line.dropFirst(4)))
            }
        }
        try process.run()
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            process.terminationHandler = { _ in c.resume() }
        }
        pipe.fileHandleForReading.readabilityHandler = nil
        try? output.value.write(to: layout.home.appendingPathComponent("install.log"), atomically: true, encoding: .utf8)
        if process.terminationStatus != 0 {
            let tail = output.value.split(separator: "\n").suffix(3).joined(separator: " ")
            throw ClientError.message("Install failed (\(process.terminationStatus)): \(tail)")
        }
    }
}

/// Starts the agent in the background and keeps it running. Restarts it after a crash, but gives up after 5 crashes in
/// 2 minutes (then the app shows the error). Exit code 3 = token revoked, 4 = another agent is already running.
final class AgentProcess {
    enum Exit { case revoked, alreadyRunning, crashedTooOften(String), stopped }

    let layout: InstallLayout
    private var process: Process?
    private var crashes: [Date] = []
    private var wantRunning = false
    var onExit: ((Exit) -> Void)?

    init(layout: InstallLayout) { self.layout = layout }

    var isOurs: Bool { process?.isRunning == true }

    func start(server: URL, token: String, relay: Bool) throws {
        wantRunning = true
        guard process?.isRunning != true else { return }
        let p = Process()
        p.executableURL = layout.python
        var args = ["-m", "syntra_agent", "--server", server.absoluteString, "--control", "--state-dir", layout.home.path,
                    "--bloombee-dir", layout.bloombeeDir.path, "--python", layout.python.path,
                    "--models-dir", layout.home.appendingPathComponent("models").path,
                    "--work-dir", layout.home.appendingPathComponent("work").path]
        if relay { args.append("--relay") }
        p.arguments = args
        p.currentDirectoryURL = layout.agentDir
        var env = ProcessInfo.processInfo.environment
        env["PYTHONPATH"] = layout.agentDir.path
        env["SYNTRA_AGENT_TOKEN"] = token  // not on the command line: other users cannot read it with ps
        p.environment = env
        FileManager.default.createFile(atPath: layout.logFile.path, contents: nil)
        let log = try FileHandle(forWritingTo: layout.logFile)
        log.seekToEndOfFile()
        p.standardOutput = log
        p.standardError = log
        p.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async { self?.handleExit(code: proc.terminationStatus, server: server, token: token, relay: relay) }
        }
        try p.run()
        process = p
    }

    private func handleExit(code: Int32, server: URL, token: String, relay: Bool) {
        process = nil
        switch code {
        case 3: wantRunning = false; onExit?(.revoked)
        case 4: onExit?(.alreadyRunning)  // a terminal agent holds the lock: the app attaches to it instead
        default:
            guard wantRunning else { onExit?(.stopped); return }
            let now = Date()
            crashes = crashes.filter { now.timeIntervalSince($0) < 120 } + [now]
            if crashes.count >= 5 {
                wantRunning = false
                onExit?(.crashedTooOften("The agent stopped \(crashes.count) times in 2 minutes (exit \(code)). See \(layout.logFile.path)."))
                return
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(crashes.count) * 2) { [weak self] in
                guard let self, self.wantRunning else { return }
                try? self.start(server: server, token: token, relay: relay)
            }
        }
    }

    /// Stop wanting it to run; the caller asks the agent to quit through the control API first (clean offline report).
    func stop() {
        wantRunning = false
        if let p = process, p.isRunning {
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { if p.isRunning { p.terminate() } }
        }
    }
}
