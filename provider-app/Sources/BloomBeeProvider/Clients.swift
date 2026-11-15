import Foundation
import IOKit
import Security

enum ClientError: LocalizedError {
    case http(Int, String)
    case notRunning
    case message(String)

    var errorDescription: String? {
        switch self {
        case .http(let code, let body):
            if let data = body.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let detail = obj["detail"] as? String ?? obj["error"] as? String {
                return detail
            }
            return "HTTP \(code)"
        case .notRunning: return "The agent is not running"
        case .message(let m): return m
        }
    }
}

private func send(_ request: URLRequest) async throws -> Data {
    let (data, response) = try await URLSession.shared.data(for: request)
    let code = (response as? HTTPURLResponse)?.statusCode ?? 0
    guard (200..<300).contains(code) else { throw ClientError.http(code, String(decoding: data, as: UTF8.self)) }
    return data
}

/// The BloomBee platform (backend): pairing, usage, sign-out.
struct PlatformClient {
    let server: URL

    func health() async throws {
        var r = URLRequest(url: server.appendingPathComponent("health"))
        r.timeoutInterval = 8
        _ = try await send(r)
    }

    func pair(email: String, password: String, machineId: String, machineName: String) async throws -> PairResponse {
        var r = URLRequest(url: server.appendingPathComponent("app/pair"))
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONSerialization.data(withJSONObject: [
            "email": email, "password": password, "machine_id": machineId, "machine_name": machineName,
        ])
        r.timeoutInterval = 15
        return try JSONDecoder.snake.decode(PairResponse.self, from: try await send(r))
    }

    func usage(token: String) async throws -> Usage {
        var r = URLRequest(url: server.appendingPathComponent("agent/usage"))
        r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        r.timeoutInterval = 10
        return try JSONDecoder.snake.decode(Usage.self, from: try await send(r))
    }

    func unpair(token: String) async throws {
        var r = URLRequest(url: server.appendingPathComponent("agent/unpair"))
        r.httpMethod = "POST"
        r.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        r.timeoutInterval = 10
        _ = try await send(r)
    }
}

/// The agent's local control API (agent/bloombee_agent/control.py). Port and secret come from ~/.bloombee/agent.json, so the
/// app also attaches to an agent that was started from a terminal.
struct ControlClient {
    let stateDir: URL

    struct Info: Decodable { let port: Int; let secret: String; let pid: Int32 }

    func info() -> Info? {
        guard let data = try? Data(contentsOf: stateDir.appendingPathComponent("agent.json")),
              let info = try? JSONDecoder().decode(Info.self, from: data), kill(info.pid, 0) == 0 else { return nil }
        return info
    }

    private func request(_ method: String, _ path: String, body: [String: Any]? = nil) async throws -> AgentStatus {
        guard let info = info() else { throw ClientError.notRunning }
        var r = URLRequest(url: URL(string: "http://127.0.0.1:\(info.port)\(path)")!)
        r.httpMethod = method
        r.setValue(info.secret, forHTTPHeaderField: "X-BloomBee-Secret")
        r.timeoutInterval = 5
        if let body {
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
            r.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        return try JSONDecoder.snake.decode(AgentStatus.self, from: try await send(r))
    }

    func status() async throws -> AgentStatus { try await request("GET", "/status") }
    func pause(now: Bool) async throws -> AgentStatus { try await request("POST", "/pause", body: ["mode": now ? "now" : "drain"]) }
    func resume() async throws -> AgentStatus { try await request("POST", "/resume") }
    func quit() async throws -> AgentStatus { try await request("POST", "/quit") }
}

/// Where the app keeps this machine's agent token: ~/.bloombee/app-tokens.json, readable only by this user (mode 0600),
/// like the agent's own control secret. (The login Keychain asked for the user's password after every rebuild of an
/// app that is not distributed through a provisioning profile.) The token is revocable on the website at any time.
enum Keychain {
    static var file: URL { FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".bloombee/app-tokens.json") }

    private static func readAll() -> [String: String] {
        guard let data = try? Data(contentsOf: file),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: String] else { return [:] }
        return obj
    }

    private static func writeAll(_ all: [String: String]) throws {
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: all, options: [.sortedKeys])
        let tmp = file.appendingPathExtension("tmp")
        FileManager.default.createFile(atPath: tmp.path, contents: data, attributes: [.posixPermissions: 0o600])
        _ = try FileManager.default.replaceItemAt(file, withItemAt: tmp)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
    }

    static func save(_ value: String, account: String) throws {
        var all = readAll()
        all[account] = value
        try writeAll(all)
    }

    static func load(account: String) -> String? { readAll()[account] }

    static func delete(account: String) {
        var all = readAll()
        all.removeValue(forKey: account)
        try? writeAll(all)
    }
}

enum Machine {
    /// Stable per Mac: the hardware UUID (IOPlatformUUID). Re-pairing the same Mac replaces its old token.
    static var id: String {
        let entry = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("IOPlatformExpertDevice"))
        defer { IOObjectRelease(entry) }
        if let uuid = IORegistryEntryCreateCFProperty(entry, "IOPlatformUUID" as CFString, kCFAllocatorDefault, 0)?
            .takeRetainedValue() as? String {
            return uuid
        }
        return Host.current().name ?? "unknown-mac"
    }

    static var name: String { Host.current().localizedName ?? "Mac" }
}
