import AppKit
import Foundation
import IOKit.ps
import ServiceManagement

@MainActor
final class AppModel: ObservableObject {
    // settings (UserDefaults); the token is in the Keychain
    @Published var serverText: String = UserDefaults.standard.string(forKey: "server") ?? "http://localhost:8000"
    @Published var email: String = UserDefaults.standard.string(forKey: "email") ?? ""
    @Published var useRelay: Bool = UserDefaults.standard.object(forKey: "relay") as? Bool ?? true
    @Published var pauseOnBattery: Bool = UserDefaults.standard.bool(forKey: "pauseOnBattery") {
        didSet { UserDefaults.standard.set(pauseOnBattery, forKey: "pauseOnBattery") }
    }

    @Published private(set) var share: ShareState = .signedOut
    @Published private(set) var status: AgentStatus?
    @Published private(set) var usage: Usage?
    @Published private(set) var lastUsageError: String?
    @Published var busy = false
    @Published var flash: String?  // one-line message under the button

    let layout = InstallLayout.standard
    private lazy var agent = AgentProcess(layout: layout)
    private lazy var control = ControlClient(stateDir: layout.home)
    private var timer: Timer?
    private var tick = 0
    private var wantShared = UserDefaults.standard.bool(forKey: "wantShared")
    private var autoPausedForBattery = false

    /// Read from the Keychain once (after the window is up) and cached: every Keychain read can make macOS ask the user.
    @Published private(set) var token: String?
    private var tokenLoaded = false
    var server: URL? { URL(string: serverText.trimmingCharacters(in: .whitespaces)) }
    var signedIn: Bool { token != nil }

    init() {
        agent.onExit = { [weak self] exit in self?.agentExited(exit) }
        share = .signedOut
        timer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.refresh() }
        }
        NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didWakeNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.flash = "Woke up: reconnecting…"; await self?.refresh(forceUsage: true) }
        }
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 300_000_000)  // let the window appear before any Keychain access
            token = Keychain.load(account: serverText)
            tokenLoaded = true
            share = signedIn ? .notSharing : .signedOut
            if signedIn && wantShared { await startSharing() } else { await refresh(forceUsage: true) }
        }
    }

    // MARK: sign in / out
    func signIn(password: String) async {
        guard let server else { flash = "Enter a valid server address"; return }
        busy = true; defer { busy = false }
        let client = PlatformClient(server: server)
        do {
            try await client.health()
        } catch {
            flash = "Cannot reach \(server.absoluteString): \(error.localizedDescription)"; return
        }
        do {
            let paired = try await client.pair(email: email, password: password, machineId: Machine.id, machineName: Machine.name)
            try Keychain.save(paired.token, account: serverText)
            token = paired.token
            UserDefaults.standard.set(serverText, forKey: "server")
            UserDefaults.standard.set(email, forKey: "email")
            UserDefaults.standard.set(useRelay, forKey: "relay")
            flash = "Signed in as \(paired.email) (\(paired.providerName))"
            share = .notSharing
            await refresh(forceUsage: true)
        } catch {
            flash = error.localizedDescription
        }
    }

    func signOut() async {
        if let token, let server { try? await PlatformClient(server: server).unpair(token: token) }  // the agent sees 401 and stops
        await stopSharing(now: true)
        Keychain.delete(account: serverText)
        token = nil
        usage = nil; status = nil
        share = .signedOut
    }

    // MARK: share / pause / stop
    func startSharing() async {
        guard let server, let token else { share = .signedOut; return }
        setWantShared(true)
        busy = true; defer { busy = false }
        if control.info() != nil {  // an agent is already running (terminal or earlier app run): attach and resume it
            _ = try? await control.resume()
            await refresh(forceUsage: true); return
        }
        if !layout.isInstalled {
            do {
                try await Installer(server: server, layout: layout).run { step in
                    Task { @MainActor in self.share = .installing(step) }
                }
            } catch {
                share = .error(error.localizedDescription); setWantShared(false); return
            }
        }
        do {
            try agent.start(server: server, token: token, relay: useRelay)
            share = .starting
        } catch {
            share = .error("Could not start the agent: \(error.localizedDescription)")
        }
    }

    func pause(now: Bool) async {
        do { status = try await control.pause(now: now); share = ShareState.derive(running: true, status: status) }
        catch { flash = error.localizedDescription }
    }

    func resume() async {
        do { status = try await control.resume(); share = ShareState.derive(running: true, status: status) }
        catch { await startSharing() }
    }

    /// Stop sharing: the GPUs go offline at once and running work moves to other GPUs.
    func stopSharing(now: Bool = true) async {
        setWantShared(false)
        _ = try? await control.quit()
        agent.stop()
        status = nil
        if signedIn { share = .notSharing }
    }

    private func setWantShared(_ v: Bool) {
        wantShared = v
        UserDefaults.standard.set(v, forKey: "wantShared")
    }

    private func agentExited(_ exit: AgentProcess.Exit) {
        switch exit {
        case .revoked:
            Keychain.delete(account: serverText)
            token = nil
            share = .signedOut
            flash = "This Mac was signed out (token revoked on the website). Sign in again to share."
        case .alreadyRunning:
            flash = "Another Syntra agent is running on this Mac; showing its status."
        case .crashedTooOften(let message):
            share = .error(message); setWantShared(false)
        case .stopped:
            if signedIn { share = .notSharing }
        }
    }

    // MARK: polling
    func refresh(forceUsage: Bool = false) async {
        guard tokenLoaded else { return }
        tick += 1
        let running = control.info() != nil
        if running {
            status = try? await control.status()
        } else {
            status = nil
        }
        if case .installing = share {} else if case .error = share, !running {} else if signedIn {
            share = running ? ShareState.derive(running: true, status: status) : (agent.isOurs ? .starting : .notSharing)
        }
        await checkBattery(running: running)
        if (forceUsage || tick % 3 == 0), let token, let server {  // usage every ~10 s
            do {
                usage = try await PlatformClient(server: server).usage(token: token)
                lastUsageError = nil
            } catch ClientError.http(401, _) {
                agentExited(.revoked)
            } catch {
                lastUsageError = error.localizedDescription
            }
        }
    }

    private func checkBattery(running: Bool) async {
        guard pauseOnBattery, running, let s = status else { return }
        let onBattery = Self.onBatteryPower()
        if onBattery && !s.paused {
            autoPausedForBattery = true
            flash = "On battery: sharing paused"
            await pause(now: false)
        } else if !onBattery && autoPausedForBattery && s.paused {
            autoPausedForBattery = false
            flash = "Back on power: sharing resumed"
            await resume()
        }
    }

    static func onBatteryPower() -> Bool {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String? else { return false }
        return type == kIOPSBatteryPowerValue
    }

    // MARK: login item
    var startsAtLogin: Bool { SMAppService.mainApp.status == .enabled }

    func setStartsAtLogin(_ on: Bool) {
        do {
            if on { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
        } catch {
            flash = "Start at login needs the app in /Applications: \(error.localizedDescription)"
        }
        objectWillChange.send()
    }

    func quitApp() {
        Task { @MainActor in
            if agent.isOurs { _ = try? await control.quit(); agent.stop() }  // a terminal-started agent keeps running
            NSApp.terminate(nil)
        }
    }

    func openLog() { NSWorkspace.shared.open(layout.logFile) }
}
